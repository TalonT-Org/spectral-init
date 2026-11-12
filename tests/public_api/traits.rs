//! Static contracts of the public types.

use spectral_init::{ComputeMode, SpectralError, SpectralInitConfig};

fn assert_send_sync_static<T: Send + Sync + 'static>() {}
fn assert_clone_partial_eq<T: Clone + PartialEq>() {}
fn assert_debug<T: std::fmt::Debug>() {}
fn assert_error<T: std::error::Error>() {}

#[test]
fn spectral_error_contracts() {
    assert_send_sync_static::<SpectralError>();
    assert_clone_partial_eq::<SpectralError>();
    assert_debug::<SpectralError>();
    assert_error::<SpectralError>();
}

#[test]
fn spectral_init_config_contracts() {
    assert_send_sync_static::<SpectralInitConfig>();
    assert_clone_partial_eq::<SpectralInitConfig>();
    assert_debug::<SpectralInitConfig>();
}

#[test]
fn compute_mode_contracts() {
    fn assert_copy_eq<T: Copy + Eq>() {}
    assert_send_sync_static::<ComputeMode>();
    assert_copy_eq::<ComputeMode>();
    assert_debug::<ComputeMode>();
}

#[test]
fn spectral_error_converts_into_a_boxed_error() {
    fn fallible() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Err(SpectralError::DegenerateEmbedding)?
    }
    let message = fallible().unwrap_err().to_string();
    assert!(message.contains("degenerate"), "{message}");
}

#[test]
fn every_error_has_a_message() {
    let errors = [
        SpectralError::DegenerateEmbedding,
        spectral_init::spectral_init(
            spectral_init::sprs::CsMatI::<f32, u32, usize>::zero((0, 0)).view(),
            None,
            &SpectralInitConfig::new(1),
        )
        .unwrap_err(),
    ];
    for error in errors {
        assert!(!error.to_string().is_empty());
        assert!(!format!("{error:?}").is_empty());
    }
}
