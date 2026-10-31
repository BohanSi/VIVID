# Third-party source and attribution

This repository vendors the user's local inference-compatible sources, excluding model weights and datasets. The local versions include VIVID-specific endpoint optical-flow export and may differ from upstream. Existing license headers and license files are retained. Non-English source annotations and runtime diagnostics were removed or rewritten in English; upstream algorithm implementations were otherwise retained.

| Directory | Upstream | License file |
| --- | --- | --- |
| RIFE | https://github.com/hzwer/ECCV2022-RIFE | [MIT](RIFE/LICENSE) |
| GIMM-VFI | https://github.com/GSeanCDAT/GIMM-VFI | [S-Lab License 1.0](GIMM-VFI/LICENSE), non-commercial use conditions |
| FILM | https://github.com/google-research/frame-interpolation | [Apache 2.0](FILM/LICENSE) |

Citations for each method are provided in the corresponding upstream README. No model weights are redistributed. Downloaded weights remain subject to their publishers' terms. Nested upstream code retains its original headers. VIVID-specific code has no new project-wide license assigned by this packaging work; the owner should choose the intended license before offering reuse permissions.
