def get_fundamentals(ticker):
    info = yf.Ticker(ticker).info
    return {
        "pe": info.get("trailingPE"),
        "eps": info.get("trailingEps"),
        "book": info.get("bookValue"),
        "fcf": info.get("freeCashflow"),
        "growth": info.get("earningsGrowth"),
        "sector": info.get("sector"),
        "margin": info.get("profitMargins")
}