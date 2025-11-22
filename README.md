# Inwesting HRP Allocator

Terminalowe narzędzie w Pythonie, które pobiera notowania z Yahoo Finance i wyznacza wagi portfela metodą Hierarchical Risk Parity (HRP).

## Wymagania

```bash
pip install -r requirements.txt
```

## Użycie

```bash
python hrp_portfolio.py --tickers SPY,IVV,AGG --period 1y --method average
```

Parametry:

- `--tickers` – lista tickerów oddzielona przecinkami (wymagana).
- `--start` / `--end` – zakres dat w formacie `YYYY-MM-DD`. Jeśli podane, parametr `--period` jest ignorowany.
- `--period` – zakres czasowy z Yahoo Finance (domyślnie `1y`).
- `--method` – metoda linkowania dla klasteryzacji hierarchicznej (`single`, `average`, `complete`, `ward`).

Program wypisze obliczone wagi HRP posortowane malejąco.
