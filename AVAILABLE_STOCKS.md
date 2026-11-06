# FinSight Stock Search

## On-demand Yahoo Finance coverage

FinSight does not maintain a permanent 60-stock universe. Users can search supported NSE/BSE ticker symbols on demand. Selected results are cached in SQLite for 24 hours.

## Previously seeded demonstration universe

The original project contained the following 60 examples, but application startup no longer generates them automatically:

## Nifty 50 Stocks (30)
1. RELIANCE - Reliance Industries
2. TCS - Tata Consultancy Services
3. INFY - Infosys
4. HDFCBANK - HDFC Bank
5. ICICIBANK - ICICI Bank
6. ADANIENT - Adani Enterprises
7. KOTAKBANK - Kotak Mahindra Bank
8. SBIN - State Bank of India
9. ITC - ITC Limited
10. BHARTIARTL - Bharti Airtel
11. LT - Larsen & Toubro
12. AXISBANK - Axis Bank
13. WIPRO - Wipro
14. HCLTECH - HCL Technologies
15. SUNPHARMA - Sun Pharma
16. BAJFINANCE - Bajaj Finance
17. MARUTI - Maruti Suzuki
18. HINDUNILVR - Hindustan Unilever
19. ASIANPAINT - Asian Paints
20. ULTRACEMCO - UltraTech Cement
21. TITAN - Titan Company
22. NTPC - NTPC Limited
23. POWERGRID - Power Grid Corporation
24. ONGC - Oil & Natural Gas Corporation
25. COALINDIA - Coal India
26. BPCL - Bharat Petroleum
27. GRASIM - Grasim Industries
28. ADANIPORTS - Adani Ports
29. DRREDDY - Dr. Reddy's Laboratories
30. EICHERMOT - Eicher Motors

## Additional Top Stocks (30)
31. TATAMOTORS - Tata Motors
32. BAJAJFINSV - Bajaj Finserv
33. TATASTEEL - Tata Steel
34. TECHM - Tech Mahindra
35. NESTLEIND - Nestle India
36. HINDALCO - Hindalco Industries
37. INDUSINDBK - IndusInd Bank
38. JSWSTEEL - JSW Steel
39. BRITANNIA - Britannia Industries
40. DIVISLAB - Divi's Laboratories
41. APOLLOHOSP - Apollo Hospitals
42. HEROMOTOCO - Hero MotoCorp
43. CIPLA - Cipla
44. TATACONSUM - Tata Consumer Products
45. SHREECEM - Shree Cement
46. UPL - UPL Limited
47. ADANIGREEN - Adani Green Energy
48. PIDILITIND - Pidilite Industries
49. SIEMENS - Siemens
50. GODREJCP - Godrej Consumer Products
51. DABUR - Dabur India
52. HAVELLS - Havells India
53. BERGEPAINT - Berger Paints
54. MARICO - Marico
55. BANDHANBNK - Bandhan Bank
56. TORNTPHARM - Torrent Pharmaceuticals
57. AMBUJACEM - Ambuja Cements
58. BOSCHLTD - Bosch Limited
59. COLPAL - Colgate-Palmolive India
60. MCDOWELL-N - United Spirits

## Features
- 90 days of historical OHLCV data per stock
- Real-time calculated metrics (RSI, MACD, PE Ratio)
- Custom FinSight Score (0-100) for each stock
- Portfolio analytics and risk assessment
- Stock analysis and technical screening

## Analyze Data

Search from the Market Analysis page or run:

```python
from data_sync import fetch_stock
history, metrics = fetch_stock('RELIANCE', 'NSE')
```
