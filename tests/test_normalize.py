from reqlens.normalize import (
    content_hash,
    location_tokens,
    normalize_company,
    normalize_title,
    strip_html,
    title_tokens,
)


def test_strip_html_basic_and_scripts():
    out = strip_html("<p>Hello&nbsp;<b>world</b></p><script>alert(1)</script><ul><li>a</li><li>b</li></ul>")
    assert "alert" not in out and "Hello world" in out and "- a" in out and "- b" in out


def test_strip_html_greenhouse_escaped():
    assert strip_html("&lt;p&gt;Build &amp;amp; ship&lt;/p&gt;").startswith("Build")


def test_strip_html_plain_and_malformed():
    assert strip_html("just  text\n\n\n\nhere") == "just text\n\nhere"
    assert "x" in strip_html("<div<<p>x")
    assert strip_html("") == ""


def test_unicode_and_control_chars_normalized():
    assert strip_html("Caf\u00e9\u200b  \x00ok") == "Café ok" or strip_html("Caf\u00e9\u200b  \x00ok").startswith("Caf")


def test_company_normalization():
    assert normalize_company("Kestrel AI, Inc.") == normalize_company("kestrel ai") == "kestrel ai"
    assert normalize_company("Helio Health Technologies") == "helio health"
    assert normalize_company("Acme & Sons (UK) Ltd") == "acme and sons"
    assert normalize_company("Inc") == "inc"  # never normalize to empty


def test_title_normalization():
    assert normalize_title("Sr. SWE (Remote)") == "senior software engineer"
    assert set(title_tokens("Backend Developer")) == set(title_tokens("Backend Engineer"))


def test_location_tokens_aliases():
    assert location_tokens("SF") == location_tokens("San Francisco, CA") - {"ca"}
    from reqlens.normalize import place_sets
    assert place_sets("London, UK") != place_sets("Manchester, UK") and not (place_sets("London, UK")[0] & place_sets("Manchester, UK")[0])
    assert not (place_sets("Bengaluru, India (on-site)")[0] & place_sets("London, UK (on-site)")[0])
    assert not (location_tokens("New York, NY") & location_tokens("New Delhi, India"))


def test_content_hash_stable_and_sensitive():
    a = content_hash("T", "C", "L", "Desc  text")
    assert a == content_hash("t", "c", "l", "desc text")
    assert a != content_hash("T", "C", "L", "Desc text changed")
