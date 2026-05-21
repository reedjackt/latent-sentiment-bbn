from data_gen.generate import generate_accounts


def test_generate_accounts_shape() -> None:
    df = generate_accounts(n=10)
    assert df.height == 10
    assert set(df.columns) == {"account_id", "domain", "page_views", "email_opens"}
