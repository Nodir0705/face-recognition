// bench.cpp — head-to-head latency benchmark for the InsightFace SCRFD detector.
//
// Measures preprocessing + ONNXRuntime inference + post-processing per frame,
// reporting p50 / p95 / p99 / mean across N iterations.
//
// Usage:
//   bench_cpp --model PATH/det_500m.onnx [--rec PATH/w600k_mbf.onnx]
//             [--image PATH | --camera N]
//             [--det-size 320] [--iters 200] [--warmup 30]
//             [--threads 2] [--score 0.5] [--nms 0.4]
//
// If --rec is supplied, each iteration also runs the recognition model on the
// largest detected face (matching what the kiosk does end-to-end). Without it
// only the detector is measured.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "pipeline.hpp"


using clock_t_ = std::chrono::high_resolution_clock;

namespace {

struct Args {
    std::string model;
    std::string rec_model;
    std::string image_path;
    int camera = -1;
    int det_size = 320;
    int iters = 200;
    int warmup = 30;
    int threads = 2;
    float score_threshold = 0.5f;
    float nms_threshold = 0.4f;
};

[[noreturn]] void die(const std::string& msg) {
    std::cerr << "bench_cpp: " << msg << "\n";
    std::exit(2);
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die("missing value for " + k);
            return argv[++i];
        };
        if      (k == "--model")    a.model = next();
        else if (k == "--rec")      a.rec_model = next();
        else if (k == "--image")    a.image_path = next();
        else if (k == "--camera")   a.camera = std::stoi(next());
        else if (k == "--det-size") a.det_size = std::stoi(next());
        else if (k == "--iters")    a.iters = std::stoi(next());
        else if (k == "--warmup")   a.warmup = std::stoi(next());
        else if (k == "--threads")  a.threads = std::stoi(next());
        else if (k == "--score")    a.score_threshold = std::stof(next());
        else if (k == "--nms")      a.nms_threshold = std::stof(next());
        else if (k == "--help" || k == "-h") {
            std::cout
                << "Usage: bench_cpp --model PATH [--rec PATH] "
                   "[--image PATH | --camera N] [--det-size 320] "
                   "[--iters 200] [--warmup 30] [--threads 2]\n";
            std::exit(0);
        }
        else die("unknown argument: " + k);
    }
    if (a.model.empty()) die("--model PATH is required (e.g. ~/.insightface/models/buffalo_sc/det_500m.onnx)");
    if (a.image_path.empty() && a.camera < 0)
        die("provide --image PATH or --camera N");
    return a;
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t idx = std::min(v.size() - 1,
                                 static_cast<size_t>(p * (v.size() - 1)));
    return v[idx];
}

cv::Mat acquire_frame(const Args& a) {
    if (!a.image_path.empty()) {
        cv::Mat img = cv::imread(a.image_path, cv::IMREAD_COLOR);
        if (img.empty()) die("could not read image: " + a.image_path);
        return img;
    }
    cv::VideoCapture cap(a.camera);
    if (!cap.isOpened()) die("could not open camera index " + std::to_string(a.camera));
    cv::Mat img;
    // Discard the first 5 frames — webcams often serve gray/dark frames during AE settle
    for (int i = 0; i < 5; ++i) cap >> img;
    cap >> img;
    if (img.empty()) die("camera read returned empty frame");
    return img;
}

// One full detect step: preprocess -> run -> decode all 3 strides -> NMS.
// Returns the detections; we don't otherwise use them in the bench (besides
// optionally feeding the largest into the rec model).
std::vector<pipeline::FaceDet> run_detect(
    Ort::Session& session,
    const std::vector<std::string>& in_names_s,
    const std::vector<std::string>& out_names_s,
    const cv::Mat& frame, int det_size,
    float score_th, float nms_th)
{
    auto pp = pipeline::preprocess_det(frame, det_size, det_size);

    std::array<int64_t, 4> shape{1, 3, det_size, det_size};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(
        mem, reinterpret_cast<float*>(pp.blob.data),
        pp.blob.total(), shape.data(), shape.size());

    std::vector<const char*> in_names, out_names;
    for (auto& s : in_names_s)  in_names.push_back(s.c_str());
    for (auto& s : out_names_s) out_names.push_back(s.c_str());

    auto outputs = session.Run(Ort::RunOptions{}, in_names.data(), &input, 1,
                                out_names.data(), out_names.size());

    // SCRFD outputs are returned in the order the model file declares them.
    // For buffalo_sc/det_500m.onnx the conventional order is:
    //   score_8, score_16, score_32,
    //   bbox_8,  bbox_16,  bbox_32,
    //   kps_8,   kps_16,   kps_32
    // We trust that order here; the daemon double-checks it by name.
    static const std::array<int, 3> strides = {8, 16, 32};
    static thread_local std::array<pipeline::AnchorCenters, 3> anchors;
    static thread_local int cached_size = -1;
    if (cached_size != det_size) {
        for (size_t i = 0; i < strides.size(); ++i) {
            anchors[i] = pipeline::make_anchors(det_size, det_size, strides[i]);
        }
        cached_size = det_size;
    }

    std::vector<pipeline::FaceDet> dets;
    dets.reserve(64);
    for (size_t i = 0; i < strides.size(); ++i) {
        const float* scores = outputs[i].GetTensorData<float>();
        const float* bboxes = outputs[i + 3].GetTensorData<float>();
        const float* kps    = outputs[i + 6].GetTensorData<float>();
        pipeline::decode_stride(scores, bboxes, kps, anchors[i],
                                 strides[i], score_th, dets);
    }
    auto kept = pipeline::nms(std::move(dets), nms_th);

    // Map detections back to original image coordinates
    for (auto& d : kept) {
        d.bbox.x /= pp.scale; d.bbox.y /= pp.scale;
        d.bbox.width /= pp.scale; d.bbox.height /= pp.scale;
        for (auto& p : d.kps) { p.x /= pp.scale; p.y /= pp.scale; }
    }
    return kept;
}

// Run the recognition model on a single 112x112 aligned face. Returns the
// (un-normalized) embedding length so we can sanity-check.
size_t run_recognize(Ort::Session& session,
                      const std::vector<std::string>& in_names_s,
                      const std::vector<std::string>& out_names_s,
                      const cv::Mat& aligned_bgr)
{
    cv::Mat blob = cv::dnn::blobFromImage(
        aligned_bgr, 1.0 / 127.5, cv::Size(112, 112),
        cv::Scalar(127.5, 127.5, 127.5), /*swapRB=*/true, /*crop=*/false, CV_32F);

    std::array<int64_t, 4> shape{1, 3, 112, 112};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(
        mem, reinterpret_cast<float*>(blob.data), blob.total(),
        shape.data(), shape.size());

    std::vector<const char*> in_names, out_names;
    for (auto& s : in_names_s)  in_names.push_back(s.c_str());
    for (auto& s : out_names_s) out_names.push_back(s.c_str());

    auto outputs = session.Run(Ort::RunOptions{}, in_names.data(), &input, 1,
                                out_names.data(), out_names.size());
    auto info = outputs[0].GetTensorTypeAndShapeInfo();
    return static_cast<size_t>(info.GetElementCount());
}

} // namespace


int main(int argc, char** argv) try {
    const Args a = parse_args(argc, argv);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "bench_cpp");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(a.threads);
    opts.SetInterOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::cout << "loading detector: " << a.model << "\n";
    Ort::Session det_session(env, a.model.c_str(), opts);
    auto det_in  = pipeline::session_input_names(det_session);
    auto det_out = pipeline::session_output_names(det_session);

    std::unique_ptr<Ort::Session> rec_session;
    std::vector<std::string> rec_in, rec_out;
    if (!a.rec_model.empty()) {
        std::cout << "loading recognizer: " << a.rec_model << "\n";
        rec_session = std::make_unique<Ort::Session>(env, a.rec_model.c_str(), opts);
        rec_in  = pipeline::session_input_names(*rec_session);
        rec_out = pipeline::session_output_names(*rec_session);
    }

    cv::Mat frame = acquire_frame(a);
    std::cout << "frame: " << frame.cols << "x" << frame.rows
              << ", det_size: " << a.det_size << "x" << a.det_size
              << ", threads: " << a.threads
              << ", warmup: " << a.warmup
              << ", iters: " << a.iters << "\n";

    // Warmup — first few iterations include allocator setup + JIT'd kernels
    for (int i = 0; i < a.warmup; ++i) {
        auto dets = run_detect(det_session, det_in, det_out, frame, a.det_size,
                                a.score_threshold, a.nms_threshold);
        if (rec_session && !dets.empty()) {
            auto& largest = *std::max_element(
                dets.begin(), dets.end(),
                [](const pipeline::FaceDet& x, const pipeline::FaceDet& y) {
                    return x.bbox.area() < y.bbox.area();
                });
            cv::Mat aligned = pipeline::align_face(frame, largest.kps);
            run_recognize(*rec_session, rec_in, rec_out, aligned);
        }
    }

    // Timed loop
    std::vector<double> det_ms, rec_ms, total_ms;
    det_ms.reserve(a.iters);
    rec_ms.reserve(a.iters);
    total_ms.reserve(a.iters);

    for (int i = 0; i < a.iters; ++i) {
        const auto t0 = clock_t_::now();
        auto dets = run_detect(det_session, det_in, det_out, frame, a.det_size,
                                a.score_threshold, a.nms_threshold);
        const auto t1 = clock_t_::now();

        double r_ms = 0.0;
        if (rec_session && !dets.empty()) {
            auto& largest = *std::max_element(
                dets.begin(), dets.end(),
                [](const pipeline::FaceDet& x, const pipeline::FaceDet& y) {
                    return x.bbox.area() < y.bbox.area();
                });
            cv::Mat aligned = pipeline::align_face(frame, largest.kps);
            const auto r0 = clock_t_::now();
            run_recognize(*rec_session, rec_in, rec_out, aligned);
            const auto r1 = clock_t_::now();
            r_ms = std::chrono::duration<double, std::milli>(r1 - r0).count();
        }
        const auto t2 = clock_t_::now();

        det_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        rec_ms.push_back(r_ms);
        total_ms.push_back(std::chrono::duration<double, std::milli>(t2 - t0).count());
    }

    auto report = [](const char* label, std::vector<double> v) {
        if (v.empty()) return;
        const double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        std::printf("  %-12s  mean=%7.2f ms  p50=%7.2f  p95=%7.2f  p99=%7.2f  min=%7.2f  max=%7.2f\n",
                     label, mean,
                     percentile(v, 0.50), percentile(v, 0.95), percentile(v, 0.99),
                     *std::min_element(v.begin(), v.end()),
                     *std::max_element(v.begin(), v.end()));
    };

    std::cout << "\nresults (n=" << a.iters << ")\n";
    report("detect", det_ms);
    if (rec_session) report("recognize", rec_ms);
    report("total", total_ms);

    // One-line machine-readable summary for scripting / Makefile diffs
    std::printf("\nSUMMARY  impl=cpp  det_p50=%.2f  det_p95=%.2f  rec_p50=%.2f  total_p50=%.2f\n",
                 percentile(det_ms, 0.50), percentile(det_ms, 0.95),
                 percentile(rec_ms, 0.50), percentile(total_ms, 0.50));
    return 0;
}
catch (const Ort::Exception& e) {
    std::cerr << "ONNXRuntime error: " << e.what() << "\n";
    return 1;
}
catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
}
