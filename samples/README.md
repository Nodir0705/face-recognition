# Sample images for benchmarking

Drop one or more face photos here as `samples/test.jpg` for the bench scripts to consume. The image isn't committed — it's gitignored so your colleagues' faces don't end up on GitHub.

A good bench image:

- Single face, ~300–500 px wide in frame
- Sharp, well-lit
- 720p or 1080p source

If you don't have one handy, any face from your phone works fine — copy it over with:

```bash
scp some_photo.jpg samples/test.jpg              # to this box
scp some_photo.jpg jarvis@192.168.3.8:~/samples/test.jpg   # to the Pi
```

The Makefile `bench-*` targets default to `samples/test.jpg` — override with `BENCH_IMAGE=path/to/other.jpg make bench-cpp`.
