from backend.features import extract, explain


def test_ip_detected():
    assert extract("http://192.168.4.5/login.php")["has_ip"] == 1


def test_brand_misuse():
    assert extract("http://paypal.verify-account.xyz/login")["brand_misuse"] == 1
    assert extract("https://www.paypal.com/signin")["brand_misuse"] == 0


def test_typosquat():
    assert extract("http://paypa1.com/")["typosquat"] == 1
    assert extract("https://google.com/")["typosquat"] == 0


def test_explain_safe():
    assert explain("https://www.wikipedia.org/")[0]["severity"] in ("good", "low")
