// pipeline.hpp — shared types + pre/post-processing for the InsightFace
// buffalo_sc model bundle, used by both bench.cpp and recognize_cpp.cpp.
//
// The detection model in buffalo_sc is SCRFD (det_500m.onnx). It outputs 9
// tensors at 3 strides (8, 16, 32) — score, bbox, kps per stride. We decode
// those, NMS them down, and (in the daemon) align each face for the recognition
// model (w600k_mbf.onnx).
//
// References:
//   * SCRFD paper: https://arxiv.org/abs/2105.04714
//   * InsightFace's reference Python decode in scrfd.py
//   * The 5-pt alignment template (arcface_dst) is a community standard.
//
// Header-only on purpose so each .cpp stays a single self-contained translation
// unit — easier to read top-to-bottom.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>


namespace pipeline {

// ----- Detection output -----

struct FaceDet {
    cv::Rect2f bbox;                    // x, y, w, h in original image coordinates
    float score;
    std::array<cv::Point2f, 5> kps;     // L_eye, R_eye, nose, L_mouth, R_mouth
};


// ----- ArcFace alignment template (112x112) -----
// These are the canonical 5 keypoint targets used by every ArcFace-derived
// recognition model (including buffalo_sc's w600k_mbf). Match Python's
// insightface/utils/face_align.py::arcface_dst exactly.
inline const std::array<cv::Point2f, 5> kArcfaceTemplate = {{
    {38.2946f, 51.6963f},  // L eye
    {73.5318f, 51.5014f},  // R eye
    {56.0252f, 71.7366f},  // nose
    {41.5493f, 92.3655f},  // L mouth
    {70.7299f, 92.2041f},  // R mouth
}};
inline constexpr int kAlignedSize = 112;


// ----- Preprocessing for the detector -----
// SCRFD wants: BGR -> RGB, normalize (x - 127.5) / 128, NCHW float32, padded to
// det_size (e.g. 320x320) preserving aspect ratio. We return the scale factor
// so we can map detection coords back to the original frame.
struct DetInput {
    cv::Mat blob;        // 1x3xHxW float32, ready for ORT
    float scale;         // multiply detection xy by 1/scale to recover original coords
    int padded_h;
    int padded_w;
};

inline DetInput preprocess_det(const cv::Mat& bgr, int det_w, int det_h) {
    const float scale = std::min(static_cast<float>(det_w) / bgr.cols,
                                 static_cast<float>(det_h) / bgr.rows);
    const int new_w = static_cast<int>(std::round(bgr.cols * scale));
    const int new_h = static_cast<int>(std::round(bgr.rows * scale));

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(new_w, new_h));

    cv::Mat padded(det_h, det_w, CV_8UC3, cv::Scalar(0, 0, 0));
    resized.copyTo(padded(cv::Rect(0, 0, new_w, new_h)));

    // blobFromImage handles BGR->RGB swap, normalization, and HWC->NCHW.
    cv::Mat blob = cv::dnn::blobFromImage(
        padded, 1.0 / 128.0, cv::Size(det_w, det_h),
        cv::Scalar(127.5, 127.5, 127.5), /*swapRB=*/true, /*crop=*/false, CV_32F);

    return {blob, scale, det_h, det_w};
}


// ----- Anchor generation for SCRFD -----
// At each stride, anchors are placed on a grid; SCRFD has 2 anchors per
// location (anchor_ratio=1.0 default). We pre-build them once per session.
struct AnchorCenters {
    std::vector<cv::Point2f> centers;  // length = grid_h * grid_w * num_anchors
};

inline AnchorCenters make_anchors(int input_h, int input_w, int stride,
                                   int num_anchors = 2) {
    AnchorCenters out;
    const int gh = input_h / stride;
    const int gw = input_w / stride;
    out.centers.reserve(gh * gw * num_anchors);
    for (int y = 0; y < gh; ++y) {
        for (int x = 0; x < gw; ++x) {
            for (int a = 0; a < num_anchors; ++a) {
                out.centers.emplace_back(static_cast<float>(x * stride),
                                         static_cast<float>(y * stride));
            }
        }
    }
    return out;
}


// ----- SCRFD output decoding -----
// For one stride: score is shape (N, 1), bbox is (N, 4) of (l,t,r,b) offsets
// in stride units, kps is (N, 10) of 5 (dx, dy) offsets in stride units.
// Distance values are multiplied by stride to get pixel offsets from anchor.
inline void decode_stride(
    const float* scores, const float* bbox_preds, const float* kps_preds,
    const AnchorCenters& anchors, int stride, float score_threshold,
    std::vector<FaceDet>& out)
{
    const size_t n = anchors.centers.size();
    for (size_t i = 0; i < n; ++i) {
        const float s = scores[i];
        if (s < score_threshold) continue;

        const cv::Point2f& a = anchors.centers[i];
        // bbox_preds layout: [l, t, r, b] distances from anchor center, in strides
        const float l = bbox_preds[i * 4 + 0] * stride;
        const float t = bbox_preds[i * 4 + 1] * stride;
        const float r = bbox_preds[i * 4 + 2] * stride;
        const float b = bbox_preds[i * 4 + 3] * stride;

        FaceDet d;
        d.score = s;
        d.bbox.x = a.x - l;
        d.bbox.y = a.y - t;
        d.bbox.width = (a.x + r) - d.bbox.x;
        d.bbox.height = (a.y + b) - d.bbox.y;

        for (int k = 0; k < 5; ++k) {
            d.kps[k].x = a.x + kps_preds[i * 10 + k * 2 + 0] * stride;
            d.kps[k].y = a.y + kps_preds[i * 10 + k * 2 + 1] * stride;
        }
        out.push_back(d);
    }
}


// ----- IoU-based NMS -----
inline float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float x1 = std::max(a.x, b.x);
    const float y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.width,  b.x + b.width);
    const float y2 = std::min(a.y + a.height, b.y + b.height);
    const float w = std::max(0.0f, x2 - x1);
    const float h = std::max(0.0f, y2 - y1);
    const float inter = w * h;
    const float uni = a.area() + b.area() - inter;
    return uni > 0 ? inter / uni : 0.0f;
}

inline std::vector<FaceDet> nms(std::vector<FaceDet> in, float iou_threshold) {
    std::sort(in.begin(), in.end(),
              [](const FaceDet& a, const FaceDet& b) { return a.score > b.score; });
    std::vector<FaceDet> kept;
    std::vector<bool> suppressed(in.size(), false);
    for (size_t i = 0; i < in.size(); ++i) {
        if (suppressed[i]) continue;
        kept.push_back(in[i]);
        for (size_t j = i + 1; j < in.size(); ++j) {
            if (suppressed[j]) continue;
            if (iou(in[i].bbox, in[j].bbox) > iou_threshold) suppressed[j] = true;
        }
    }
    return kept;
}


// ----- 5-point similarity-transform alignment -----
// Solves the 2D similarity transform that maps the detected 5 landmarks onto
// the canonical arcface template, then warps the original image into a 112x112
// crop. Identical to InsightFace Python's `norm_crop`.
inline cv::Mat align_face(const cv::Mat& bgr,
                           const std::array<cv::Point2f, 5>& kps,
                           float scale_back = 1.0f)
{
    std::vector<cv::Point2f> src(kps.begin(), kps.end());
    if (scale_back != 1.0f) {
        for (auto& p : src) { p.x /= scale_back; p.y /= scale_back; }
    }
    std::vector<cv::Point2f> dst(kArcfaceTemplate.begin(), kArcfaceTemplate.end());
    // estimateAffinePartial2D uses RANSAC by default — for 5 points with no
    // outliers we want LMEDS or no robust method at all. cv::LMEDS works well.
    cv::Mat M = cv::estimateAffinePartial2D(src, dst, cv::noArray(), cv::LMEDS);
    cv::Mat aligned;
    cv::warpAffine(bgr, aligned, M, cv::Size(kAlignedSize, kAlignedSize),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return aligned;
}


// ----- L2 normalize a vector in-place -----
inline void l2_normalize(std::vector<float>& v) {
    double sum = 0.0;
    for (float x : v) sum += static_cast<double>(x) * x;
    const float n = static_cast<float>(std::sqrt(sum));
    if (n > 1e-12f) for (auto& x : v) x /= n;
}


// ----- Cosine match against a row-major gallery of L2-normalized embeddings -----
// Returns (best_index, best_similarity). best_index = -1 if below threshold.
inline std::pair<int, float> match(const std::vector<float>& probe,
                                    const std::vector<float>& gallery,
                                    int dim, float threshold)
{
    const int n = static_cast<int>(gallery.size() / dim);
    if (n == 0 || probe.size() != static_cast<size_t>(dim)) return {-1, 0.0f};
    int best = -1;
    float best_sim = -1.0f;
    for (int i = 0; i < n; ++i) {
        const float* row = gallery.data() + i * dim;
        float s = 0.0f;
        for (int k = 0; k < dim; ++k) s += row[k] * probe[k];
        if (s > best_sim) { best_sim = s; best = i; }
    }
    return (best_sim >= threshold) ? std::make_pair(best, best_sim)
                                    : std::make_pair(-1, best_sim);
}


// ----- Tiny convenience: get all input/output names from an ORT session -----
inline std::vector<std::string> session_input_names(Ort::Session& s) {
    Ort::AllocatorWithDefaultOptions a;
    std::vector<std::string> out;
    for (size_t i = 0; i < s.GetInputCount(); ++i) {
        auto p = s.GetInputNameAllocated(i, a);
        out.emplace_back(p.get());
    }
    return out;
}

inline std::vector<std::string> session_output_names(Ort::Session& s) {
    Ort::AllocatorWithDefaultOptions a;
    std::vector<std::string> out;
    for (size_t i = 0; i < s.GetOutputCount(); ++i) {
        auto p = s.GetOutputNameAllocated(i, a);
        out.emplace_back(p.get());
    }
    return out;
}

} // namespace pipeline
