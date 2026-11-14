from core.paper_config import load_paper_config

_CLASSIC = load_paper_config()["classic_baselines"]
BASELINE_PROTOCOLS = {
    method: int(spec["protocol"]) for method, spec in _CLASSIC.items()
}
PATCHNETVLAD_PROTOCOL = BASELINE_PROTOCOLS["Patch-NetVLAD"]
