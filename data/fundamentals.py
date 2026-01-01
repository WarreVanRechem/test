import yfinance as yf


def get_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        info = {}

    return {
        "pe": info.get("trailingPE"),
        "eps": info.get("trailingEps"),
        "book": info.get("bookValue"),
        "fcf": info.get("freeCashflow"),
        "growth": info.get("earningsGrowth"),
        "sector": info.get("sector"),
        "margin": info.get("profitMargins"),
    }
