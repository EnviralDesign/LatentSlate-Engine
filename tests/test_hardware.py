from latentslate_engine.hardware import (
    architecture_name,
    capability_metadata,
    sm_number,
    supports_fp8,
    supports_nvfp4,
)


def test_blackwell_capability_metadata():
    metadata = capability_metadata((12, 0))

    assert sm_number((12, 0)) == 120
    assert metadata["sm"] == "sm120"
    assert metadata["architecture"] == "Blackwell or newer"
    assert metadata["capabilities"] == {"fp8": True, "nvfp4": True}


def test_non_blackwell_keeps_portable_capabilities():
    assert architecture_name((8, 9)) == "Ada Lovelace"
    assert supports_fp8((8, 9)) is True
    assert supports_nvfp4((8, 9)) is False
    assert architecture_name((8, 6)) == "Ampere"
    assert supports_fp8((8, 6)) is False
    assert supports_nvfp4((8, 6)) is False
