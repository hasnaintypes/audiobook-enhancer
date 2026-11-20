from audiobook_engine.enhancement.resemble import resolve_enhance_params


def test_resolve_enhance_params_denoise_only_prioritizes_noise_removal():
    params = resolve_enhance_params(denoise_only=True)

    assert params["lambd"] == 0.9


def test_resolve_enhance_params_enhance_prioritizes_naturalness():
    params = resolve_enhance_params(denoise_only=False)

    assert params["lambd"] == 0.1


def test_resolve_enhance_params_fixed_values():
    params = resolve_enhance_params(denoise_only=False)

    assert params["nfe"] == 32
    assert params["solver"] == "midpoint"
    assert params["tau"] == 0.5
