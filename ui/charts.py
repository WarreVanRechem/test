import plotly.graph_objects as go
from plotly.subplots import make_subplots

def price_chart(df):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3]
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Prijs"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df['SMA200'], name="SMA 200"),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df['RSI'], name="RSI"),
        row=2, col=1
    )

    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    return fig
