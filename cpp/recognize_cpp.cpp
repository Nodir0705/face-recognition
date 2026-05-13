// recognize_cpp.cpp — C++ recognition daemon.
//
// Equivalent to src/recognize.py: owns the camera, runs SCRFD detection +
// ArcFace embeddings, matches against a gallery loaded from the same SQLite
// `attendance.db` the Python web app uses, and writes IN/OUT events back to
// the same DB. You can run this INSTEAD of the Python recognition thread
// (don't run both at once — they'd fight over the camera).
//
// Why this exists alongside Python:
//   * No GIL — detection runs truly parallel with the Flask MJPEG threads
//     (when those still serve previews; this daemon doesn't serve HTTP).
//   * Tight inner loop, no per-call Python dispatch.
//   * Honest A/B latency comparison vs the Python implementation.
//
// What this does NOT include (yet):
//   * The mask/sunglasses occlusion heuristic (port from src/occlusion.py)
//   * The blink/motion liveness gate (port from src/liveness.py)
//   * The MJPEG preview server (the Python web app keeps that role)
//
// Both omissions are intentional scope cuts so the daemon fits in one file.
// They're easy to add later — the helper signatures in pipeline.hpp are stable.
//
// Usage:
//   recognize_cpp --models DIR --db PATH [--config PATH]
//                 [--camera N] [--det-size 320] [--threads 2]
//                 [--threshold 0.42] [--cooldown 300]
//
//   --models is the directory containing det_500m.onnx + w600k_mbf.onnx
//            (typically ~/.insightface/models/buffalo_sc/)
//   --db     points at data/attendance.db relative to the project root
//   --config is optional; if given, we read attendance/recognition fields from
//            it (simple grep — we don't pull in yaml-cpp). CLI args override.

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <ctime>
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <sqlite3.h>

#include "pipeline.hpp"


using clock_t_ = std::chrono::high_resolution_clock;

namespace {

// ---------- CLI ----------

struct Args {
    std::string models_dir;
    std::string db_path;
    std::string config_path;
    int camera = 0;
    int det_size = 320;
    int threads = 2;
    float threshold = 0.42f;
    int min_face_px = 80;
    int consecutive = 3;
    int cooldown_sec = 300;
    bool toggle_mode = true;
    int day_start_hour = 4;
    float det_score = 0.5f;
    float nms = 0.4f;
};

[[noreturn]] void die(const std::string& msg) {
    std::cerr << "recognize_cpp: " << msg << "\n";
    std::exit(2);
}

// Tiny YAML-ish reader: just looks for "key: value" or "  key: value" lines
// under known sections. Good enough for our config; no need for yaml-cpp.
std::optional<std::string> get_yaml_scalar(const std::string& path,
                                            const std::string& section,
                                            const std::string& key)
{
    std::ifstream f(path);
    if (!f) return std::nullopt;
    std::string line, current;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (!std::isspace(static_cast<unsigned char>(line[0]))) {
            // Top-level key like "recognition:"
            auto colon = line.find(':');
            if (colon != std::string::npos) current = line.substr(0, colon);
            continue;
        }
        if (current != section) continue;
        // Indented: "  key: value"
        size_t i = 0;
        while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) ++i;
        if (line.compare(i, key.size(), key) != 0) continue;
        size_t after = i + key.size();
        if (after >= line.size() || line[after] != ':') continue;
        ++after;
        while (after < line.size() && std::isspace(static_cast<unsigned char>(line[after]))) ++after;
        // Strip trailing comment
        auto hash = line.find('#', after);
        std::string val = line.substr(after, hash == std::string::npos ? std::string::npos : hash - after);
        // Trim
        while (!val.empty() && std::isspace(static_cast<unsigned char>(val.back()))) val.pop_back();
        // Strip surrounding quotes
        if (val.size() >= 2 && val.front() == '"' && val.back() == '"')
            val = val.substr(1, val.size() - 2);
        return val;
    }
    return std::nullopt;
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die("missing value for " + k);
            return argv[++i];
        };
        if      (k == "--models")       a.models_dir = next();
        else if (k == "--db")           a.db_path = next();
        else if (k == "--config")       a.config_path = next();
        else if (k == "--camera")       a.camera = std::stoi(next());
        else if (k == "--det-size")     a.det_size = std::stoi(next());
        else if (k == "--threads")      a.threads = std::stoi(next());
        else if (k == "--threshold")    a.threshold = std::stof(next());
        else if (k == "--min-face-px")  a.min_face_px = std::stoi(next());
        else if (k == "--consecutive")  a.consecutive = std::stoi(next());
        else if (k == "--cooldown")     a.cooldown_sec = std::stoi(next());
        else if (k == "--score")        a.det_score = std::stof(next());
        else if (k == "--nms")          a.nms = std::stof(next());
        else if (k == "--help" || k == "-h") {
            std::cout << "Usage: recognize_cpp --models DIR --db PATH [--config PATH] "
                          "[--camera 0] [--det-size 320] [--threads 2] "
                          "[--threshold 0.42] [--cooldown 300]\n";
            std::exit(0);
        }
        else die("unknown argument: " + k);
    }
    if (a.models_dir.empty()) die("--models DIR required");
    if (a.db_path.empty())    die("--db PATH required");

    if (!a.config_path.empty()) {
        if (auto v = get_yaml_scalar(a.config_path, "recognition", "match_threshold"))
            a.threshold = std::stof(*v);
        if (auto v = get_yaml_scalar(a.config_path, "recognition", "min_face_px"))
            a.min_face_px = std::stoi(*v);
        if (auto v = get_yaml_scalar(a.config_path, "recognition", "consecutive_frames"))
            a.consecutive = std::stoi(*v);
        if (auto v = get_yaml_scalar(a.config_path, "attendance", "cooldown_sec"))
            a.cooldown_sec = std::stoi(*v);
        if (auto v = get_yaml_scalar(a.config_path, "attendance", "day_start_hour"))
            a.day_start_hour = std::stoi(*v);
        if (auto v = get_yaml_scalar(a.config_path, "attendance", "toggle_mode"))
            a.toggle_mode = (*v == "true" || *v == "True" || *v == "1");
    }
    return a;
}


// ---------- SQLite helpers ----------
// Mirror the schema in src/db.py exactly. We don't run CREATE TABLE here —
// the Python app already did that.

class DB {
public:
    explicit DB(const std::string& path) {
        if (sqlite3_open(path.c_str(), &db_) != SQLITE_OK)
            die(std::string("sqlite open failed: ") + sqlite3_errmsg(db_));
        exec("PRAGMA journal_mode=WAL");
        exec("PRAGMA foreign_keys=ON");
        exec("PRAGMA busy_timeout=5000");
    }
    ~DB() { if (db_) sqlite3_close(db_); }

    void exec(const char* sql) {
        char* err = nullptr;
        if (sqlite3_exec(db_, sql, nullptr, nullptr, &err) != SQLITE_OK) {
            std::string msg = err ? err : "unknown";
            sqlite3_free(err);
            die("sqlite exec '" + std::string(sql) + "': " + msg);
        }
    }

    // Load every active employee's embeddings into a flat row-major matrix.
    // emp_ids/names are length-N parallel arrays describing each row.
    void load_gallery(std::vector<float>& gallery,
                       std::vector<std::string>& emp_ids,
                       std::vector<std::string>& names,
                       int dim = 512)
    {
        gallery.clear(); emp_ids.clear(); names.clear();
        sqlite3_stmt* st = nullptr;
        const char* sql = "SELECT emp_id, name, embedding FROM employees WHERE active = 1";
        if (sqlite3_prepare_v2(db_, sql, -1, &st, nullptr) != SQLITE_OK)
            die(std::string("prepare: ") + sqlite3_errmsg(db_));
        while (sqlite3_step(st) == SQLITE_ROW) {
            std::string emp = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
            std::string nm  = reinterpret_cast<const char*>(sqlite3_column_text(st, 1));
            const void* blob = sqlite3_column_blob(st, 2);
            int bytes = sqlite3_column_bytes(st, 2);
            const float* fp = static_cast<const float*>(blob);
            const int rows = bytes / (dim * static_cast<int>(sizeof(float)));
            for (int r = 0; r < rows; ++r) {
                gallery.insert(gallery.end(), fp + r * dim, fp + (r + 1) * dim);
                emp_ids.push_back(emp);
                names.push_back(nm);
            }
        }
        sqlite3_finalize(st);
    }

    struct LastEvent { std::string type; long long ts; bool exists; };

    LastEvent last_event(const std::string& emp_id) {
        sqlite3_stmt* st = nullptr;
        const char* sql = "SELECT event_type, timestamp FROM attendance "
                          "WHERE emp_id = ? ORDER BY timestamp DESC LIMIT 1";
        sqlite3_prepare_v2(db_, sql, -1, &st, nullptr);
        sqlite3_bind_text(st, 1, emp_id.c_str(), -1, SQLITE_TRANSIENT);
        LastEvent r{"", 0, false};
        if (sqlite3_step(st) == SQLITE_ROW) {
            r.type   = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
            r.ts     = sqlite3_column_int64(st, 1);
            r.exists = true;
        }
        sqlite3_finalize(st);
        return r;
    }

    long long log_event(const std::string& emp_id, const std::string& event_type,
                         double confidence)
    {
        sqlite3_stmt* st = nullptr;
        const char* sql = "INSERT INTO attendance (emp_id, event_type, timestamp, "
                          "confidence, synced) VALUES (?, ?, ?, ?, 0)";
        sqlite3_prepare_v2(db_, sql, -1, &st, nullptr);
        const long long now = std::time(nullptr);
        sqlite3_bind_text  (st, 1, emp_id.c_str(),     -1, SQLITE_TRANSIENT);
        sqlite3_bind_text  (st, 2, event_type.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int64 (st, 3, now);
        sqlite3_bind_double(st, 4, confidence);
        if (sqlite3_step(st) != SQLITE_DONE)
            die(std::string("log_event: ") + sqlite3_errmsg(db_));
        long long rid = sqlite3_last_insert_rowid(db_);
        sqlite3_finalize(st);
        return rid;
    }

private:
    sqlite3* db_ = nullptr;
};


// ---------- IoU tracker (mirrors src/recognize.py::Tracker) ----------

struct Track {
    cv::Rect2f bbox;
    int missed = 0;
    std::deque<std::string> match_history;   // empty string = unmatched
};

class Tracker {
public:
    explicit Tracker(float iou_th = 0.3f, int max_missed = 8)
        : iou_th_(iou_th), max_missed_(max_missed) {}

    std::vector<std::pair<int, size_t>> update(const std::vector<pipeline::FaceDet>& dets) {
        std::vector<std::pair<int, size_t>> assignments;  // (track_id, det_index)
        std::set<int> used;
        for (size_t i = 0; i < dets.size(); ++i) {
            int best_id = -1;
            float best_iou = 0.0f;
            for (auto& [tid, tr] : tracks_) {
                if (used.count(tid)) continue;
                float u = pipeline::iou(dets[i].bbox, tr.bbox);
                if (u > best_iou && u > iou_th_) { best_iou = u; best_id = tid; }
            }
            if (best_id < 0) {
                best_id = next_id_++;
                tracks_[best_id] = Track{dets[i].bbox, 0, {}};
            } else {
                tracks_[best_id].bbox = dets[i].bbox;
                tracks_[best_id].missed = 0;
            }
            used.insert(best_id);
            assignments.emplace_back(best_id, i);
        }
        // Increment missed and prune stale tracks
        for (auto it = tracks_.begin(); it != tracks_.end();) {
            if (!used.count(it->first)) {
                if (++it->second.missed > max_missed_) it = tracks_.erase(it);
                else ++it;
            } else ++it;
        }
        return assignments;
    }

    Track* get(int id) {
        auto it = tracks_.find(id);
        return it == tracks_.end() ? nullptr : &it->second;
    }

private:
    float iou_th_;
    int max_missed_;
    std::unordered_map<int, Track> tracks_;
    int next_id_ = 0;
};


// ---------- IN/OUT decision (mirrors src/recognize.py::decide_event_type) ----------

long long today_start_unix(int day_start_hour) {
    std::time_t now = std::time(nullptr);
    std::tm tm = *std::localtime(&now);
    tm.tm_hour = day_start_hour;
    tm.tm_min = 0;
    tm.tm_sec = 0;
    if (std::localtime(&now)->tm_hour < day_start_hour) tm.tm_mday -= 1;
    return static_cast<long long>(std::mktime(&tm));
}

std::optional<std::string> decide_event(DB& db, const std::string& emp_id,
                                         const Args& a)
{
    auto le = db.last_event(emp_id);
    const long long now = std::time(nullptr);
    if (le.exists && (now - le.ts) < a.cooldown_sec) return std::nullopt;
    if (a.toggle_mode) {
        const long long ts0 = today_start_unix(a.day_start_hour);
        if (!le.exists || le.ts < ts0) return std::string("IN");
        return std::string(le.type == "IN" ? "OUT" : "IN");
    } else {
        std::tm tm = *std::localtime(&now);
        return std::string(tm.tm_hour < 12 ? "IN" : "OUT");
    }
}


// ---------- Detection inference (same as bench.cpp::run_detect) ----------

std::vector<pipeline::FaceDet> detect(
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
        mem, reinterpret_cast<float*>(pp.blob.data), pp.blob.total(),
        shape.data(), shape.size());

    std::vector<const char*> in_names, out_names;
    for (auto& s : in_names_s)  in_names.push_back(s.c_str());
    for (auto& s : out_names_s) out_names.push_back(s.c_str());

    auto outputs = session.Run(Ort::RunOptions{}, in_names.data(), &input, 1,
                                out_names.data(), out_names.size());

    static const std::array<int, 3> strides = {8, 16, 32};
    static thread_local std::array<pipeline::AnchorCenters, 3> anchors;
    static thread_local int cached_size = -1;
    if (cached_size != det_size) {
        for (size_t i = 0; i < strides.size(); ++i)
            anchors[i] = pipeline::make_anchors(det_size, det_size, strides[i]);
        cached_size = det_size;
    }

    std::vector<pipeline::FaceDet> dets;
    for (size_t i = 0; i < strides.size(); ++i) {
        const float* scores = outputs[i].GetTensorData<float>();
        const float* bboxes = outputs[i + 3].GetTensorData<float>();
        const float* kps    = outputs[i + 6].GetTensorData<float>();
        pipeline::decode_stride(scores, bboxes, kps, anchors[i], strides[i],
                                 score_th, dets);
    }
    auto kept = pipeline::nms(std::move(dets), nms_th);
    for (auto& d : kept) {
        d.bbox.x /= pp.scale; d.bbox.y /= pp.scale;
        d.bbox.width /= pp.scale; d.bbox.height /= pp.scale;
        for (auto& p : d.kps) { p.x /= pp.scale; p.y /= pp.scale; }
    }
    return kept;
}


// ---------- Embedding inference ----------

std::vector<float> embed(Ort::Session& session,
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
    const size_t n = info.GetElementCount();
    const float* p = outputs[0].GetTensorData<float>();
    std::vector<float> v(p, p + n);
    pipeline::l2_normalize(v);
    return v;
}


// ---------- Camera (no-op wrapper around cv::VideoCapture) ----------

class Camera {
public:
    explicit Camera(int index) : cap_(index) {
        if (!cap_.isOpened()) die("could not open camera index " + std::to_string(index));
        // 1280x720 to match the Python config's default
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    }
    bool read(cv::Mat& out) { return cap_.read(out); }
private:
    cv::VideoCapture cap_;
};


// ---------- Signal handling for clean shutdown ----------

std::atomic<bool> g_running{true};
extern "C" void on_signal(int) { g_running.store(false); }

} // namespace


int main(int argc, char** argv) try {
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    const Args a = parse_args(argc, argv);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "recognize_cpp");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(a.threads);
    opts.SetInterOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    const std::string det_path = a.models_dir + "/det_500m.onnx";
    const std::string rec_path = a.models_dir + "/w600k_mbf.onnx";

    std::cout << "loading det:  " << det_path << "\n";
    Ort::Session det_session(env, det_path.c_str(), opts);
    auto det_in  = pipeline::session_input_names(det_session);
    auto det_out = pipeline::session_output_names(det_session);

    std::cout << "loading rec:  " << rec_path << "\n";
    Ort::Session rec_session(env, rec_path.c_str(), opts);
    auto rec_in  = pipeline::session_input_names(rec_session);
    auto rec_out = pipeline::session_output_names(rec_session);

    DB db(a.db_path);
    std::vector<float> gallery;
    std::vector<std::string> emp_ids, names;
    db.load_gallery(gallery, emp_ids, names);
    const int dim = 512;
    std::cout << "gallery: " << (gallery.size() / dim) << " embeddings, "
              << std::set<std::string>(emp_ids.begin(), emp_ids.end()).size()
              << " employees\n";
    auto last_reload = std::time(nullptr);
    constexpr int RELOAD_EVERY = 60;

    Camera cam(a.camera);
    Tracker tracker;

    cv::Mat frame;
    while (g_running.load()) {
        if (!cam.read(frame) || frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        // Periodic gallery reload — picks up new enrollments from the web UI
        if (std::time(nullptr) - last_reload > RELOAD_EVERY) {
            db.load_gallery(gallery, emp_ids, names);
            last_reload = std::time(nullptr);
        }

        auto dets_all = detect(det_session, det_in, det_out, frame, a.det_size,
                                a.det_score, a.nms);
        // Filter tiny faces (matches src/recognize.py)
        std::vector<pipeline::FaceDet> dets;
        dets.reserve(dets_all.size());
        for (auto& d : dets_all) {
            if (d.bbox.width >= a.min_face_px) dets.push_back(d);
        }

        auto assignments = tracker.update(dets);

        for (auto [tid, di] : assignments) {
            cv::Mat aligned = pipeline::align_face(frame, dets[di].kps);
            auto e = embed(rec_session, rec_in, rec_out, aligned);
            auto [idx, sim] = pipeline::match(e, gallery, dim, a.threshold);

            Track* tr = tracker.get(tid);
            if (!tr) continue;

            if (idx < 0) {
                tr->match_history.push_back("");
                if (tr->match_history.size() > 10) tr->match_history.pop_front();
                continue;
            }
            const std::string& emp = emp_ids[idx];
            tr->match_history.push_back(emp);
            if (tr->match_history.size() > 10) tr->match_history.pop_front();

            // Need N consecutive matches to the same person
            const int n = static_cast<int>(tr->match_history.size());
            if (n < a.consecutive) continue;
            bool stable = true;
            for (int i = n - a.consecutive; i < n; ++i) {
                if (tr->match_history[i] != emp || tr->match_history[i].empty()) {
                    stable = false; break;
                }
            }
            if (!stable) continue;

            auto evt = decide_event(db, emp, a);
            if (!evt) continue;
            long long rid = db.log_event(emp, *evt, sim);
            std::cout << "[#" << rid << "] " << emp << " " << names[idx]
                      << " -> " << *evt << " (sim=" << sim << ")\n";
            tr->match_history.clear();
        }
    }

    std::cout << "shutting down\n";
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
