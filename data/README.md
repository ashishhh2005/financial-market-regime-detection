# Data

The project generates a reproducible synthetic financial-market dataset locally.
No proprietary or real client/bank data is included.

```bash
python src/generate_data.py
```

This creates `data/market_data.csv` (2,500 trading days, 25 volume values deliberately left blank for the cleaning step).
The file is not committed to Git because it can be regenerated exactly from the fixed random seed.
