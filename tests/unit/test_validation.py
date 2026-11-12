from audiobook_engine.ingestion.validation import SourceKind, classify_source


def test_standard_watch_url_is_video():
    result = classify_source("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert result.kind is SourceKind.VIDEO
    assert result.is_supported


def test_short_url_is_video():
    result = classify_source("https://youtu.be/dQw4w9WgXcQ")
    assert result.kind is SourceKind.VIDEO


def test_playlist_url_is_playlist():
    result = classify_source(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL1234567890"
    )
    assert result.kind is SourceKind.PLAYLIST


def test_non_youtube_host_is_unsupported():
    result = classify_source("https://vimeo.com/12345")
    assert not result.is_supported
    assert result.reason is not None


def test_empty_url_is_unsupported():
    result = classify_source("")
    assert not result.is_supported


def test_malformed_video_id_is_unsupported():
    result = classify_source("https://www.youtube.com/watch?v=short")
    assert not result.is_supported
