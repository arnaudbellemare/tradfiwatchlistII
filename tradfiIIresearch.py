import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import logging
from scipy.stats import linregress, chi2
from scipy.linalg import solve_toeplitz
import cvxpy as cp
from arch import arch_model
from sklearn.covariance import LedoitWolf
from functools import lru_cache
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression, Ridge
from tenacity import retry, stop_after_attempt, wait_exponential, wait_fixed, retry_if_exception_type
import random
from pandas_datareader import data as pdr
import plotly.graph_objects as go
from statsmodels.tsa.api import Holt
from sklearn.preprocessing import RobustScaler
import statsmodels.tsa.stattools as smt

# --- Basic Configuration ---
logging.basicConfig(filename='stock_analysis.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
st.set_page_config(page_title="Quantitative Portfolio Analysis", layout="wide")

# --- Data Structures and Mappings ---
# (Keeping your existing data structures)
sector_etf_map = {
    'Technology': 'XLK', 'Consumer Cyclical': 'XLY', 'Communication Services': 'XLC',
    'Financial Services': 'XLF', 'Industrials': 'XLI', 'Basic Materials': 'XLB',
    'Energy': 'XLE', 'Real Estate': 'XLRE', 'Healthcare': 'XLV',
    'Consumer Defensive': 'XLP', 'Utilities': 'XLU'
}
# Add key factor ETFs for new features
factor_etfs = ['QQQ', 'IWM', 'DIA', 'EEM', 'EFA', 'IVE', 'IVW', 'MDY', 'MTUM', 'RSP', 'SPY', 'QUAL', 'SIZE', 'USMV']
etf_list = list(set(sector_etf_map.values()) | set(factor_etfs))
default_weights = {
    "(Dividends + Share Buyback) / FCF": 5.0, "CapEx / (Depr + Amor)": 4.5, "Debt Ratio": 6.0,
    "Gross Profit Margin": 7.5, "Inventory Turnover": 4.5, "Net Profit Margin": 6.5,
    "Return on Assets": 6.0, "Assets Growth TTM": 5.5, "Assets Growth QOQ": 7.0,
    "Assets Growth YOY": 5.5, "FCF Growth TTM": 5.0, "FCF Growth QOQ": 6.0,
    "FCF Growth YOY": 6.0, "Dividend Yield": 4.0, "FCF Yield": 6.5, "Operating Margin": 4.5,
    "Liabilities to Equity Ratio": 4.5, "Earnings Per Share, Diluted": 4.5, "Dividend Payout Ratio": 0,
    "Return On Invested Capital": 6.0, "Piotroski F-Score": 6.5, "Operating Leverage": 5.5,
    "Cash Return On Invested Capital": 6.0, "Asset Turnover": 4.5, "Current Ratio": 6.0,
    "Dividends / FCF": 5.5, "Interest Coverage": 2.5, "Quick Ratio": 4.5, "Return on Equity": 7.0,
    "Share Buyback / FCF": 5.5, "Earnings Growth TTM": 5.5, "Earnings Growth QOQ": 6.5,
    "Earnings Growth YOY": 6.5, "Sales Growth TTM": 5.5, "Sales Growth QOQ": 5.5,
    "Sales Growth YOY": 6.5, "Earnings Yield": 6.5, "Market-Cap": 4.5, "P/E": 4.5, "P/Sales": 4.5,
    "Free Cash Flow": 3.5, "Free Cash Flow to Net Income": 5.5, "Sales Per Share": 4.5,
    "Free Cash Flow Per Share": 5.5, "Sharpe Ratio": 18.0, "Relative Z-Score": 18.0,
    "Market Correlation": 4.5, "Correlation_Score": 4.5, "Trend": 5.5, "Q Score": 6.5,
    "Coverage Score": 5.5, "Beta_to_SPY": 4.5, "GARCH_Vol": 5.5, "Vision": 4.5,
    "Value Factor": 6.0, "Profitability_Factor": 6.0, "Log_Log_Utility": 5.0,
    "Vol_Autocorr": 5.0, "Log_Log_Sharpe": 10.0, "Stop_Loss_Impact": 2.5, "AR_Coeff": 4.0,
    "Tangible_Book_Value": 4.5, "Return_On_Tangible_Equity": 6.0, "Insider_Ownership_Ratio": 6.0,
    "Earnings_Growth_Rate_5y": 5.5, "Peter_Lynch_Fair_Value": 4.5, "Peter_Lynch_Fair_Value_Upside": 6.0,
    "Revenue_Growth_Rate_5y": 5.5, "Meets_Triple_Rule": 3.0, "Return_21d": 4.0, "Return_63d": 4.5,
    "Return_126d": 5.0, "Return_252d": 5.5, "Audit Risk": 3.0, "Board Risk": 3.0, "Compensation Risk": 3.0,
    "Shareholder Rights Risk": 3.0, "Overall Risk": 4.0, "Institutional Ownership Ratio": 6.0,
    "Hurst Exponent (Lo's R/S)": 6.0
}
columns = [
    "Ticker", "Name","Market_Cap", "Dividend_Yield", "PE_Ratio", "EPS_Diluted",
    "Sales_Per_Share", "FCF_Per_Share", "Asset_Turnover", "CapEx_to_DepAmor",
    "Current_Ratio", "Debt_Ratio", "Dividends_to_FCF", "Gross_Profit_Margin",
    "Interest_Coverage", "Inventory_Turnover", "Net_Profit_Margin", "Quick_Ratio",
    "ROA", "ROE", "Share_Buyback_to_FCF", "Dividends_Plus_Buyback_to_FCF",
    "Assets_Growth_TTM", "Earnings_Growth_TTM", "FCF_Growth_TTM", "Sales_Growth_TTM",
    "Assets_Growth_QOQ", "Earnings_Growth_QOQ", "FCF_Growth_QOQ", "Sales_Growth_QOQ",
    "Assets_Growth_YOY", "Earnings_Growth_YOY", "FCF_Growth_YOY", "Sales_Growth_YOY",
    "FCF_Yield", "RD_to_Gross_Profit_2Y_Avg", "Earnings_Yield", "PS_Ratio", "Free_Cash_Flow", "Operating_Margin",
    "FCF_to_Net_Income", "Liabilities_to_Equity", "Dividend_Payout_Ratio",
    "Operating_Leverage", "Piotroski_F-Score", "ROIC", "Cash_ROIC",
    "Dollar_Volume_90D", "Score", "Sharpe_Ratio", "Relative_Z_Score",
    "Rolling_Market_Correlation", "Correlation_Score", "Trend", "Q_Score",
    "Coverage_Score", "Risk_Flag", "Beta_to_SPY", "GARCH_Vol", "Vision",
    "Best_Factor", "Value_Factor", "Profitability_Factor", "Log_Log_Utility",
    "Vol_Autocorr", "Log_Log_Sharpe", "Stop_Loss_Impact", "AR_Coeff", "Sector", "Tangible_Book_Value", "Return_On_Tangible_Equity",
    "Insider_Ownership_Ratio", "Earnings_Growth_Rate_5y",
    "Peter_Lynch_Fair_Value", "Peter_Lynch_Fair_Value_Upside",
    "Revenue_Growth_Rate_5y", "Meets_Triple_Rule", "Return_21d", "Return_63d", "Return_126d", "Return_252d",
    "Audit_Risk", "Board_Risk", "Compensation_Risk", "Shareholder_Rights_Risk",
    "Overall_Risk", "Institutional_Ownership_Ratio", "Hurst_Exponent",
    "Momentum", "Growth" # Added for new factor demonstrations
]
tickers = ["KLAC", "NVDA", "NEU", "APH", "MSFT", "UI", "CME", "HLI", "EXEL", "HWM", "LRCX", "APP", "SCCO", "NFLX","DJT", "LLY", "FBTC", "EHC", "VRSK", "MCO", "RMD", "ALLE", "CTAS", "META", "CRS", "AGX", "INTU", "HALO", "RCL", "HEI", "CF", "GOOG", "EME", "DRI", "WWD", "TT", "TMUS", "BR", "BAH", "FTDR", "TDG", "FCFS", "CBOE", "VST", "JNJ", "AVGO", "BLK", "EAT", "NTAP", "IESC", "COP", "MTDR", "AIT", "AMAT", "SFM", "CW", "TPR", "STX", "RL", "HIMS", "WTS", "HCA", "NDAQ", "VRT", "AMGN", "COST", "ESE", "ROL", "MPLX", "FAST", "NVT", "CTRA", "BMI", "COR", "MELI", "WING", "NEM", "NSC", "FIX", "LRN", "BWXT", "EXPE", "POWL", "WELL", "SXT", "MA", "UTHR", "GD", "USFD", "PM", "ETN", "QDEL", "EVR", "MDT", "VEEV", "BKNG", "WINA", "FICO", "FSS", "FTNT", "PATK", "LECO", "GILD", "RSG", "MCK", "ICE", "CAKE", "PWR", "PLTR", "ALSN", "IPAR", "HESM", "GWW", "SYY", "ITW", "AWI", "PKG", "IBM", "QCOM", "CSCO", "CAH", "ITT", "LII", "DPZ", "URI", "TXRH", "TXN", "MO", "GSHD", "EBAY", "AJG", "FTI", "MSI", "ZTS", "CTRE", "IMO", "IDXX", "ORCL", "ITRI", "DY", "V", "GRMN", "PPC", "SPGI", "LEU", "UBER", "ALV", "LNG", "ADP", "SNPS", "TGLS", "GE", "PRIM", "UNP", "BSY", "MWA", "AXON", "TRGP", "EA", "ABT", "PH", "WAB", "FFIV", "JCI", "LIN", "VRSN", "MPWR", "TEL", "HD", "MAR", "MORN", "CACC", "NYT", "UHS", "QLYS", "SSNC", "CPRX", "LCII", "CL", "JBL", "OVV", "BX", "PJT", "YUM", "CVNA", "NXST", "FOX", "RRR", "DTE", "NTES", "WM", "LMT", "ACN", "DGX", "ROP", "MRK", "UGI", "BYD", "CR", "XOM", "DUOL", "MCD", "NDSN", "KTB", "LAD", "WMB", "APO", "MGY", "REVG", "ADI", "FN", "DE", "DLB", "GEN", "MNST", "HLT", "SBAC", "ENB", "ADT", "SYK", "CTSH", "EW", "BRO", "FLR", "CHTR", "VZ", "ARMK", "PLNT", "EXPD", "PNR", "PAYX", "MSA", "CHE", "GGG", "CDNS", "T", "AON", "CDW", "NRG", "IDCC", "CAT", "PTC", "TNL", "DIS", "UNH", "BRBR", "ATI", "DOCS", "LPLA", "JHG", "BJ", "SANM", "CCK", "ANSS", "PAG", "PR", "AMD", "XEL", "TW", "PIPR", "ATMU", "VMI", "WEC", "KMI", "RTX", "LULU", "HUBB","CORZ", "SHW", "OSK","RDDT", "PCAR", "CPA", "RS", "CACI", "AZO", "AROC", "PINS", "SNEX", "DORM", "PTCT", "MSCI", "PODD", "ECL", "ROK", "WDFC", "JKHY", "MTN", "HAL", "NOW", "TROW", "IRDM", "STE", "FDS", "MLI", "SBRA", "EXE", "PAYC", "HURN", "SCI", "GDDY", "BABA", "TSCO", "EMR", "ESAB", "CSL", "VICI", "DG", "SNA", "TDY", "GVA", "ORLY", "CHEF", "BMRN", "CIVI", "CHDN", "DK", "FHI", "EXR", "SU", "ENS", "MC", "HOOD", "DELL", "NXPI", "VRTX", "HON", "HAS", "UFPT", "KRYS", "CMCSA", "BDC", "ENSG", "LVS", "ULTA", "PG", "HRB", "ISRG", "SPXC", "GLPI", "ABBV", "EOG", "TER", "KR", "VMC", "PEGA", "HES", "GFF", "HSY", "SSD", "CPRT", "CEG", "UPWK", "CROX", "ATR", "EPD", "DCI", "HQY", "LOPE", "CORT", "TJX", "BSX", "BCPC", "ELS", "FLO", "H", "G", "DKS", "AEIS", "COKE", "EQT", "ROAD", "AMP", "VC", "HAE", "CVX", "WSM", "AAPL", "DDS", "REGN", "INGR", "WH", "MEDP", "CMG", "JAZZ", "AMT", "AEE", "PEP", "ET", "NFG", "LAZ", "KFY", "OSIS", "PCTY", "DOV", "WDC", "LAMR", "MMSI", "NTNX", "ZBH", "IT", "GHC", "WMT", "PHM", "ATO", "ACMR", "MATX", "TKR", "MAIN", "OXY", "WCC", "NEE", "TTC", "OKE", "QSR", "HLNE", "TRU", "CRM", "BKR", "IBP", "AYI", "CNM", "EXP", "URBN", "SKYW", "SWKS", "DT", "AME", "WMG", "BOOT", "FLEX", "WES", "ELF", "F", "PBH", "MANH", "AOS", "OTIS", "APPF", "CPNG", "MKSI", "OLED", "XYL", "GPOR", "SLB", "TMO", "TTMI", "ADM", "TYL", "AMZN", "PEG", "EPR", "STAG", "FANG", "PAGP", "SMLR", "NOC", "CWH", "ADBE", "DECK", "TXT", "LOW", "NBIX", "AMCR", "MSM", "CVLT", "K", "CNP", "YUMC", "SPSC", "CRH", "A", "BRK-A", "LNTH", "MKTX", "RPRX", "CSX", "PII", "SAIC", "NET", "ADSK", "GMS", "CARR", "RMBS", "THC", "IVZ", "CTVA", "CRUS", "PAA", "NNN", "CBT", "AR", "WMS", "CHH", "WPC", "MMS", "ES", "TDW", "FDX", "OHI", "KMB", "WCN", "NKE", "POWI", "WAT", "CMI", "CLX", "LOGI", "OTEX", "BMY", "FIVE", "UPS", "YOU", "BBWI", "EXC", "PPL", "NUE", "EFX", "STRL", "AMG", "MPC", "WSO", "DXCM", "CVS", "IEX", "WHD", "DVN", "TRMB", "KAI", "AVY", "WST", "TMDX", "GM", "CIEN", "UAL", "NSP", "HXL", "ZM", "YELP", "KVUE", "TTD", "LNT", "FOUR", "MAT", "FLS", "ROST", "PSX", "RPM", "KEX", "LKQ", "CP", "LTH", "CARG", "DDOG", "CGNX", "BROS", "ANF", "LHX", "PFE", "NVR", "OC", "TMHC", "MARA", "PANW", "AL", "GTLS", "GNTX", "MIDD", "GPN", "EMN", "GLW", "AVT", "PEN", "INFA", "BCO", "STZ", "HST", "CBRE", "FTAI", "ASO", "THO", "D", "TRNO", "OWL", "ZBRA", "RRC", "NI", "CWEN", "BLD", "ABNB", "CHD", "ACIW", "SKX", "POOL", "SKY", "KO", "VAC", "ED", "DVA", "SLGN", "GPC", "RKLB", "AMED", "PSN", "IQV", "FUL", "WFRD", "LH", "CLH", "LPX", "MUSA", "FE", "SMCI", "VIRT", "GEHC", "MMM", "GTES", "LEA", "TDC", "MAS", "ACM", "GNRC", "CVCO", "DOCU", "DBX", "ESS", "COHR", "KDP", "PPG", "DOX", "GPI", "BDX", "VNT", "GIS", "FNF", "PGNY", "OPCH", "BOX", "UDR", "MOH", "RBA", "JLL", "SHOO", "SBUX", "EGP", "BXP", "MLM", "CRC", "TGTX", "MTD", "ODFL", "UFPI", "APA", "MYRG", "CELH", "FI", "VNOM", "GPK", "PARA", "TEX", "RVLV", "BRKR", "KEYS", "OMC", "CDE", "JBHT", "BIIB", "FCX", "SRE", "ACAD", "MDLZ", "CMS", "INCY", "MTZ", "OGN", "ST", "SPG", "BALL", "EPRT", "LFUS", "ATGE", "CUBE", "INSM", "HTHT", "GATX", "IP", "J", "EXLS", "VLO", "HOLX", "XPO", "MTCH", "DAL", "ALKS", "LYB", "ENTG", "SLM", "BL", "ARRY", "CNC", "LW", "FMC", "ELAN", "ARW", "KNF", "YETI", "CAVA", "ESI", "SITE", "TREX", "WEN", "ABG", "VNO", "TGT", "MKC", "HP", "RRX", "BFAM", "DHR", "AVB", "MHO", "R", "TTEK", "SEE", "CNXC", "CRI", "NOVT", "MGM", "AA", "TOST", "BG", "KIM", "MSTR", "SMPL", "LEVI", "REG", "VVV", "PINC", "WSC", "MSGS", "ROIV", "NVS", "ETR", "SMG", "TGNA", "HRI", "ZS", "VRNA", "CCJ", "ALGN", "AMH", "BLDR", "DX", "STLD", "SWX", "LYV", "PSTG", "INOD", "TGI", "PK", "HPE", "BEN", "LYFT", "BZ", "SON", "CHWY", "SHAK", "WDAY", "SO", "CBZ", "EVRG", "ROOT", "FRT", "AN", "GXO", "CHRW", "PYPL", "HR", "KHC", "CCL", "HL", "TCOM", "GDS", "MTSI", "LBRT", "VSEC", "COO", "WYNN", "HOG", "AXTA", "HRL", "INVH", "SR", "GWRE", "HSIC", "FTV", "BCC", "MGNI", "ACLS", "ZTO", "CCCS", "IDA", "BBY", "APTV", "MOG-A", "PZZA", "EQR", "MAA", "FND", "RBLX", "SKT", "LIVN", "DRS", "SWK", "BURL", "ARCC", "KBR", "SNX", "ON", "FIS", "VSCO", "WU", "CSGP", "AAOI", "EPAM", "TFX", "TAP", "BKH", "MOS", "SYNA", "JEF", "SM", "AKAM", "TSLA", "POST", "NSIT", "GOLF", "WHR", "TEAM", "ITGR", "RGC", "VOD", "AAON", "COIN", "FSLR", "DV", "ALNY", "SIG", "BC", "TECH", "BWA", "SAND", "DUK", "TNET", "M", "TRP", "TPH", "PLAY", "IRM", "MOD", "MAC", "TSSI", "NCLH", "NFE", "HGV", "QRVO", "BILL", "QTWO", "IPG", "REXR", "ETSY", "NE", "DASH", "CRGY", "INSP", "BE", "COTY", "PLD", "TMC", "AFRM", "DEI", "CBRL", "SMR", "EG", "DXC", "LSTR", "TIGR", "NJR", "CAG", "DAN", "AEO", "TSN", "HII", "COLM", "WY", "FCN", "KNX", "KRG", "HPQ", "ORA", "CRWD", "FR", "SOFI", "WRBY", "CLSK", "EIX", "OLN", "RYTM", "IONS", "RGLD", "CUZ", "WTRG", "AMKR", "GMED", "SNOW", "BRX", "ABR", "AXSM", "FROG", "WTW", "KTOS", "TDS", "VCYT", "OKTA", "DKNG", "APD", "ICLR", "AVDX", "O", "LSCC", "CRK", "RHI", "CNK", "IONQ", "EQIX", "BPMC", "GME", "CPB", "JXN", "SAIA", "B", "ENPH", "RIOT", "PTGX", "EYE", "APLE", "FUN", "MQ", "SIRI", "TTWO", "KSS", "TARS", "SAP", "BTU", "PNW", "ASGN", "NOK", "KRC", "WW", "GO", "MNDY", "SJM", "SYM", "VRNS", "AEM", "MHK", "MP", "VITL", "KKR", "PVH", "BBIO", "OGS", "KGC", "GEO", "NSA", "MASI", "CAR", "ABM", "GT", "MDB", "AMR", "CPT", "DAR", "APLD", "LITE", "IRTC", "FIVN", "LUV", "AVA", "HASI", "ARQT", "AVTR", "PRGO", "VTR", "HUBS", "CNX", "MCHP", "ROKU", "RXRX", "CENX", "ALK", "AVAV", "NVAX", "SPB", "DD", "CCI", "FRPT", "FUBO", "DOC", "EQH", "NOG", "AES", "PCG", "WBA", "TENB", "EL", "SMTC", "CUK", "PCOR", "OUST", "Z", "ARE", "SAM", "MIR", "XRAY", "CC", "BCRX", "ADMA", "SLAB", "ACHC", "DBRG", "NLY", "SOUN", "TWLO", "RDNT", "GRPN", "TNXP", "DOW", "CRL", "DINO", "BRZE", "BA", "RGTI", "PL", "UPST", "AGCO", "DLTR", "WK", "AKRO", "KYMR", "VERX", "TEVA", "UUUU", "ESTC", "PGY", "PI", "CE", "GH", "GTLB", "IR", "AAL", "STEP", "SGRY", "HSAI", "VTRS", "ICUI", "STLA", "ILMN", "COMP", "AEVA", "PRCT", "ACHR", "MRVL", "CPRI", "HCTI", "PARR", "NVST", "ALKT", "CWAN", "HTZ", "CYTK", "COMM", "MRCY", "TDOC", "CZR", "AMBA", "IFF", "SNAP", "QBTS", "ALGM", "CALX", "KD", "FL", "ZETA", "SLNO", "GKOS", "INTA", "BEAM", "LUMN", "AI", "ALC", "DNB", "RNA", "HCC", "NOV", "PTEN", "MRUS", "W", "CABO", "RNG", "FRSH", "CCOI", "PAAS", "MBLY", "SATS", "SITM", "KMX", "CFLT", "ENVX", "ALHC", "MAN", "MPW", "TVTX", "SG", "IREN", "INTC", "VSAT", "S", "CRNX", "BAX", "NCNO", "RDW", "CLF", "IOVA", "SUI", "CIFR", "PCT", "PATH", "FOLD", "PTON", "NGD", "RVMD", "AG", "VFC", "MUR", "TWST", "RIG", "AGI", "VERA", "FYBR", "DYN", "GENI", "ALB", "NUVL", "HUN", "ACVA", "ASH", "IRT", "CRSP", "EXAS", "NWL", "QS", "PENN", "SRPT", "ZLAB", "SPR", "ALIT", "JOBY", "RGEN", "WBD", "LCID", "IAC", "NVTS", "AAP", "MDGL", "VERV", "HBI", "APVO", "NIO", "TNDM", "VKTX", "APLS", "ACLX", "RELY", "RUN", "PCVX", "RKT", "U", "AMC", "RARE", "DOCN", "NEOG", "AUR", "TXG", "RHP", "OPEN", "CNR", "LUNR", "JBLU", "ASAN", "RIVN", "COLD", "BB", "ESLT", "NTLA", "DAVE", "JPM", "SEDG", "BTI", "GFL", "TME", "CRDO", "MRNA", "WFC", "CASY", "CLS", "QUBT", "TBBK", "PLUG", "SAN", "FUTU", "MS", "GS", "DB", "PFGC", "SEIC", "SRAD", "STT", "RXO", "MAG", "BMO", "NTRS", "FSK", "CCEP", "TIGO", "C", "IBKR", "COF", "ING", "ATAT", "RCAT", "EWBC", "URTH", "BBAI", "WBS", "BPOP", "BK", "OLLI", "BCS", "MT", "TGB", "SCHW", "BBVA", "UNM", "SF", "APG", "MKL", "OMF", "VNET", "SYF", "TCBI", "RYAAY", "AWK", "NVMI", "L", "RJF", "FERG", "BFH", "GRAB", "STWD", "ONB", "OZK", "UBS", "GRRR", "RNST", "GFI", "LYG", "TSM", "BAP", "BUD", "FNB", "NXT", "FHN", "RF", "CFG", "AER", "RITM", "STNE", "HBAN", "DTM", "ERJ", "FLUT", "CALM", "TTE", "CADE", "EQNR", "ZION", "QFIN", "GRND", "KEY", "GNW", "RY", "MTB", "ASND", "CBSH", "ITUB", "RELX", "TS", "CRBG", "LMND", "NWG", "WTFC", "QGEN", "MLGO", "SE", "ASB", "WRB", "CX", "CFR", "ARES", "PNFP", "HUT", "AXP", "SNV", "WULF", "VOYA", "SPHR", "XPEV", "HIG", "HDB", "PNC", "SHEL", "BKU", "BAM", "AXS", "GSK", "BILI", "WGS", "FCNCA", "NGG", "HMC", "BAC", "VLY", "TPL", "TAK", "CMA", "LDOS", "BTBT", "ALLY", "HWC", "UEC", "CINF", "HOLO", "TFC", "UL", "EOSE", "PLMR", "ACI", "FITB", "MUFG", "YMM", "SSRM", "HIW", "OGE", "MMYT", "USB", "DNN", "KC", "CYBR", "AIG", "UBSI", "IBN", "GLNG", "LNC", "TRV", "BOH", "GL", "CAMT", "BTDR", "MU", "STM", "WAL", "UMBF", "BNS", "PGR", "TPG", "CB", "ARGX", "AZN", "SLG", "SMFG", "THG", "WTM", "HMY", "CHKP", "CNO", "ORI", "ALL", "ZWS", "LEGN", "NTRA", "UMC", "KNSL", "AGO", "NU", "ONON", "AMX", "AIZ", "SSB", "PB", "BANC", "VAL", "MTG", "XP", "ZIM", "AEP", "PFSI", "PBF", "SNY", "MULN", "EH", "ABEV", "MET", "ACGL", "OSCR", "AFL", "AFG", "CHX", "RGA", "GFS", "GGB", "FRO", "WIX", "GBCI", "RIO", "CI", "TAL", "GMAB", "SIGI", "ERIE", "COLB", "PECO", "VIST", "PFG", "POR", "STNG", "NICE", "BHF", "WEX", "FMX", "AUB", "HUM", "MMC", "PBR-A", "TM", "SQM", "RNR", "ARR", "CWST", "RLI", "BIO", "JHX", "CMC", "ADC", "HLN", "PRU", "ESNT", "KMPR", "KNTK", "DHI", "DSGX", "DEO", "EEFT", "INFY", "ATKR", "GLBE", "KBH", "PSA", "JD", "WLK", "LEN", "RH", "FAF", "ELV", "PRGS", "GLOB", "AGNC", "ANET", "TRIP", "SRRK", "ASTS", "XENE", "TOL", "MTH", "BHVN", "SMMT"]
#tickers = ["KLAC", "NVDA", "NEU", "APH", "MSFT", "UI", "CME", "HLI", "EXEL", "HWM", "LRCX", "APP", "SCCO", "NFLX", "LLY", "EHC", "VRSK", "MCO", "RMD", "ALLE", "CTAS", "META", "CRS", "AGX", "INTU", "HALO", "RCL", "HEI", "CF", "GOOG", "EME", "DRI", "WWD", "TT", "TMUS", "BR", "BAH", "FTDR", "TDG", "FCFS", "CBOE", "VST", "JNJ", "AVGO", "BLK", "EAT", "NTAP", "IESC", "COP", "MTDR", "AIT", "AMAT", "SFM", "CW", "TPR", "STX", "RL", "HIMS", "WTS", "HCA", "NDAQ", "VRT", "AMGN", "COST", "ESE", "ROL", "MPLX", "FAST", "NVT", "CTRA", "BMI", "COR", "MELI", "WING", "NEM", "NSC", "FIX", "LRN", "BWXT", "EXPE", "POWL", "WELL", "SXT", "MA", "UTHR", "GD", "USFD", "PM", "ETN", "QDEL", "EVR", "MDT", "VEEV", "BKNG", "WINA", "FICO", "FSS", "FTNT", "PATK", "LECO", "GILD", "RSG", "MCK", "ICE", "CAKE", "PWR", "PLTR", "ALSN", "IPAR", "HESM", "GWW", "SYY", "ITW", "AWI", "PKG", "IBM", "QCOM", "CSCO", "CAH", "ITT", "LII", "DPZ", "URI", "TXRH", "TXN", "MO", "GSHD", "EBAY", "AJG", "FTI", "MSI", "ZTS", "CTRE", "IMO", "IDXX", "ORCL", "ITRI", "DY", "V", "GRMN", "PPC", "SPGI", "LEU", "UBER", "ALV", "LNG", "ADP", "SNPS", "TGLS", "GE", "PRIM", "UNP", "BSY", "MWA", "AXON", "TRGP", "EA", "ABT", "PH", "WAB", "FFIV", "JCI", "LIN", "VRSN", "MPWR", "TEL", "HD", "MAR", "MORN", "CACC", "NYT", "UHS", "QLYS", "SSNC", "CPRX", "LCII", "CL", "JBL", "OVV", "BX", "PJT", "YUM", "CVNA", "NXST", "FOX", "RRR", "DTE", "NTES", "WM", "LMT", "ACN", "DGX", "ROP", "MRK", "UGI", "BYD", "CR", "XOM", "DUOL", "MCD", "NDSN", "KTB", "LAD"]
METRIC_NAME_MAP = {
    "Dividends_Plus_Buyback_to_FCF": "(Dividends + Share Buyback) / FCF", "CapEx_to_DepAmor": "CapEx / (Depr + Amor)",
    "ROA": "Return on Assets", "ROE": "Return on Equity", "ROIC": "Return On Invested Capital",
    "Cash_ROIC": "Cash Return On Invested Capital", "PS_Ratio": "P/Sales", "EPS_Diluted": "Earnings Per Share, Diluted",
    "Free_Cash_Flow": "Free Cash Flow", "FCF_to_Net_Income": "Free Cash Flow to Net Income", "Sales_Per_Share": "Sales Per Share",
    "FCF_Per_Share": "Free Cash Flow Per Share", "Piotroski_F-Score": "Piotroski F-Score", "PE_Ratio": "P/E",
    "Dividend_Yield": "Dividend Yield", "FCF_Yield": "FCF Yield", "Operating_Margin": "Operating Margin",
    "Liabilities_to_Equity": "Liabilities to Equity Ratio", "Dividend_Payout_Ratio": "Dividend Payout Ratio",
    "Operating_Leverage": "Operating Leverage", "Assets_Growth_TTM": "Assets Growth TTM", "Earnings_Growth_TTM": "Earnings Growth TTM",
    "FCF_Growth_TTM": "FCF Growth TTM", "Sales_Growth_TTM": "Sales Growth TTM", "Assets_Growth_QOQ": "Assets Growth QOQ",
    "Earnings_Growth_QOQ": "Earnings Growth QOQ", "FCF_Growth_QOQ": "FCF Growth QOQ", "Sales_Growth_QOQ": "Sales Growth QOQ",
    "Assets_Growth_YOY": "Assets Growth YOY", "Earnings_Growth_YOY": "Earnings Growth YOY", "FCF_Growth_YOY": "FCF Growth YOY",
    "Sales_Growth_YOY": "Sales Growth YOY", "Earnings_Yield": "Earnings Yield", "Market_Cap": "Market-Cap", "Debt_Ratio": "Debt Ratio",
    "Gross_Profit_Margin": "Gross Profit Margin", "Inventory_Turnover": "Inventory Turnover", "Net_Profit_Margin": "Net Profit Margin",
    "Asset_Turnover": "Asset Turnover", "Current_Ratio": "Current Ratio", "Dividends_to_FCF": "Dividends / FCF",
    "Interest_Coverage": "Interest Coverage", "Quick_Ratio": "Quick Ratio", "Share_Buyback_to_FCF": "Share Buyback / FCF",
    "Sharpe_Ratio": "Sharpe Ratio", "Relative_Z_Score": "Relative Z-Score", "Rolling_Market_Correlation": "Market Correlation",
    "Correlation_Score": "Correlation_Score", "Trend": "Trend", "Q_Score": "Q Score", "Coverage_Score": "Coverage Score",
    "Risk_Flag": "Risk_Flag", "Beta_to_SPY": "Beta_to_SPY", "GARCH_Vol": "GARCH_Vol", "Vision": "Vision", "Best_Factor": "Best_Factor",
    "Value_Factor": "Value Factor", "Profitability_Factor": "Profitability Factor", "Log_Log_Utility": "Log_Log_Utility",
    "Vol_Autocorr": "Vol_Autocorr", "Log_Log_Sharpe": "Log_Log_Sharpe", "Stop_Loss_Impact": "Stop_Loss_Impact",
    "AR_Coeff": "AR_Coeff", "Tangible_Book_Value": "Tangible Book Value", "Return_On_Tangible_Equity": "Return on Tangible Equity",
    "Insider_Ownership_Ratio": "Insider Ownership Ratio", "Earnings_Growth_Rate_5y": "5-Year Earnings Growth Rate",
    "Peter_Lynch_Fair_Value": "Peter Lynch Fair Value", "Peter_Lynch_Fair_Value_Upside": "Peter Lynch Fair Value Upside",
    "Revenue_Growth_Rate_5y": "5-Year Revenue Growth Rate", "Meets_Triple_Rule": "Meets Triple Rule", "Return_21d": "21-Day Return",
    "Return_63d": "63-Day Return", "Return_126d": "126-Day Return", "Return_252d": "252-Day Return", "Audit_Risk": "Audit Risk",
    "Board_Risk": "Board Risk", "Compensation_Risk": "Compensation Risk", "Shareholder_Rights_Risk": "Shareholder Rights Risk",
    "Overall_Risk": "Overall Risk", "Institutional_Ownership_Ratio": "Institutional Ownership Ratio",
    "Hurst_Exponent": "Hurst Exponent (Lo's R/S)", "Momentum": "Momentum", "Growth": "Growth"
}
REVERSE_METRIC_NAME_MAP = {v: k for k, v in METRIC_NAME_MAP.items()}


################################################################################
# SECTION 1: ALL FUNCTION DEFINITIONS
################################################################################

# --- Helper Functions ---
def calculate_growth(current, previous):
    if pd.isna(current) or pd.isna(previous) or previous == 0: return np.nan
    return (current - previous) / abs(previous) * 100

def metric_name(col):
    return METRIC_NAME_MAP.get(col, col)

def get_value(df, possible_keys, col_index=0):
    for key in possible_keys:
        if key in df.index and df.loc[key] is not None:
            series = df.loc[key]
            if isinstance(series, pd.Series) and len(series) > col_index:
                val = series.iloc[col_index]
                return pd.to_numeric(val, errors='coerce') if val is not None else np.nan
            elif not isinstance(series, pd.Series):
                 val = series
                 return pd.to_numeric(val, errors='coerce') if val is not None else np.nan
    return np.nan

# --- SVD-based PSD Matrix Correction (Section 9.5) ---
def nearest_psd_matrix(matrix):
    """
    Ensure a matrix is positive semi-definite using SVD for numerical stability.
    """
    try:
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)
        # Symmetrize the matrix
        matrix = (matrix + matrix.T) / 2

        # Perform SVD
        U, s, Vt = np.linalg.svd(matrix, full_matrices=False)

        # Reconstruct with non-negative singular values
        s = np.maximum(s, 1e-10)
        psd_matrix = U @ np.diag(s) @ Vt

        # Symmetrize again to ensure exact symmetry after reconstruction
        psd_matrix = (psd_matrix + psd_matrix.T) / 2
        return psd_matrix
    except Exception as e:
        logging.error(f"Error in PSD correction: {e}")
        return np.eye(len(matrix)) # Fallback to identity matrix


# --- Deep Dive Data Functions (for UI) ---
# (Keeping your existing deep dive functions)
@st.cache_data
def fetch_and_organize_deep_dive_data(_ticker_symbol):
    try:
        ticker = yf.Ticker(_ticker_symbol)
        info = ticker.info
        hist_10y = ticker.history(period="10y")
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        cashflow = ticker.cashflow
        if not info: return {"Error": f"Could not retrieve info for {_ticker_symbol}."}
        price_data = {
            'Price': info.get('currentPrice', info.get('regularMarketPrice')), 'Change': info.get('regularMarketChange'),
            'Change (%)': info.get('regularMarketChangePercent', 0) * 100, 'Day Low': info.get('dayLow'),
            'Day High': info.get('dayHigh'), 'Year High': info.get('fiftyTwoWeekHigh'), 'Year Low': info.get('fiftyTwoWeekLow'),
            '50-Day Avg': info.get('fiftyDayAverage'), '200-Day Avg': info.get('twoHundredDayAverage'),
            'Exchange': info.get('exchange'), 'Volume': info.get('volume'), 'Avg Volume': info.get('averageVolume'),
            'Open': info.get('open'), 'Previous Close': info.get('previousClose'), 'Market Cap': info.get('marketCap'),
            'Shares Outstanding': info.get('sharesOutstanding'), 'Beta': info.get('beta'),
            'Enterprise Value': info.get('enterpriseValue')
        }
        perf_data = {}
        if not hist_10y.empty:
            close_prices = hist_10y['Close']
            periods = {'1D': 1, '5D': 5, '1M': 21, '3M': 63, '6M': 126, 'YTD': None, '1Y': 252, '3Y': 252*3, '5Y': 252*5, '10Y': 252*10, 'Max': len(close_prices)-1}
            for name, p in periods.items():
                try:
                    if name == 'YTD':
                        ytd_start_price = close_prices[close_prices.index.year == datetime.now().year].iloc[0]
                        perf_data[name] = (close_prices.iloc[-1] / ytd_start_price - 1) * 100 if ytd_start_price > 0 else np.nan
                    elif p is not None and len(close_prices) > p and p > 0:
                        start_price = close_prices.iloc[-(p+1)]
                        perf_data[name] = (close_prices.iloc[-1] / start_price - 1) * 100 if start_price > 0 else np.nan
                    elif p is None and name == 'Max' and len(close_prices) > 1:
                        start_price = close_prices.iloc[0]
                        perf_data[name] = (close_prices.iloc[-1] / start_price - 1) * 100 if start_price > 0 else np.nan
                    else: perf_data[name] = np.nan
                except IndexError: perf_data[name] = np.nan
        ratios_ttm = {
            'P/E Ratio (TTM)': info.get('trailingPE'), 'Forward P/E Ratio': info.get('forwardPE'),
            'P/S Ratio (TTM)': info.get('priceToSalesTrailing12Months'), 'P/B Ratio (TTM)': info.get('priceToBook'),
            'EV/Revenue (TTM)': info.get('enterpriseToRevenue'), 'EV/EBITDA (TTM)': info.get('enterpriseToEbitda'),
            'Earnings Yield (TTM)': (1 / info.get('trailingPE')) * 100 if info.get('trailingPE') and info.get('trailingPE') != 0 else np.nan,
            'FCF Yield (TTM)': (info.get('freeCashflow', 0) / info.get('marketCap', 1)) * 100 if info.get('marketCap') and info.get('marketCap') != 0 else np.nan,
            'Dividend Yield (TTM)': info.get('dividendYield', 0) * 100,
            'Payout Ratio (TTM)': info.get('payoutRatio'), 'Current Ratio (TTM)': info.get('currentRatio'),
            'Quick Ratio (TTM)': info.get('quickRatio'), 'Debt/Equity (TTM)': info.get('debtToEquity'),
            'Return on Equity (ROE, TTM)': info.get('returnOnEquity', 0) * 100,
            'Return on Assets (ROA, TTM)': info.get('returnOnAssets', 0) * 100,
            'Gross Margin (TTM)': info.get('grossMargins', 0) * 100,
            'Operating Margin (TTM)': info.get('operatingMargins', 0) * 100,
            'Profit Margin (TTM)': info.get('profitMargins', 0) * 100,
        }
        def statement_to_df(df):
            if df is None or df.empty: return pd.DataFrame({"Data Not Available": []})
            df_display = df.copy()
            df_display.index.name = "Metric"
            df_display.columns = [d.strftime('%Y-%m-%d') for d in df_display.columns]
            return df_display.applymap(lambda x: f'{x:,.0f}' if isinstance(x, (int, float)) else x)
        return {
            "Price Data": price_data, "Performance": perf_data, "Key Ratios (TTM)": ratios_ttm,
            "Income Statement": statement_to_df(financials), "Balance Sheet": statement_to_df(balance_sheet),
            "Cash Flow": statement_to_df(cashflow),
        }
    except Exception as e: return {"Error": f"An error occurred: {e}"}

def display_deep_dive_data(ticker_symbol):
    data = fetch_and_organize_deep_dive_data(ticker_symbol)
    if "Error" in data:
        st.error(data["Error"])
        return
    for section, content in data.items():
        with st.expander(f"**{section}**", expanded=(section == "Key Ratios (TTM)")):
            if isinstance(content, dict):
                df = pd.DataFrame.from_dict(content, orient='index', columns=['Value'])
                df.index.name = 'Metric'
                df['Value'] = df['Value'].apply(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)
                st.dataframe(df, use_container_width=True)
            elif isinstance(content, pd.DataFrame):
                st.dataframe(content, use_container_width=True)

# --- Advanced Metric & Data Fetching Functions ---
@lru_cache(maxsize=None)
def fetch_etf_history(ticker, period="3y"):
    history = yf.Ticker(ticker).history(period=period, auto_adjust=True, interval="1d")
    if history.empty or 'Close' not in history.columns: raise ValueError(f"No valid data for {ticker}")
    history.index = history.index.tz_localize(None)
    history.dropna(subset=['Close'], inplace=True)
    if history['Close'].eq(0).any(): raise ValueError(f"Zero Close prices for {ticker}")
    return history

@st.cache_data
def fetch_all_etf_histories(_etf_list, period="3y"):
    etf_histories = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_etf = {executor.submit(fetch_etf_history, etf, period): etf for etf in _etf_list}
        for future in tqdm(as_completed(future_to_etf), total=len(_etf_list), desc="Fetching ETF Histories"):
            etf = future_to_etf[future]
            try: etf_histories[etf] = future.result()
            except Exception as e: logging.error(f"Failed to fetch ETF history for {etf}: {e}")
    return etf_histories

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_ticker_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    history = ticker.history(period="3y", auto_adjust=True, interval="1d").tz_localize(None)
    info = ticker.info
    financials = ticker.financials
    balancesheet = ticker.balance_sheet
    cashflow = ticker.cashflow
    quarterly_financials = ticker.quarterly_financials
    quarterly_balancesheet = ticker.quarterly_balance_sheet
    quarterly_cashflow = ticker.quarterly_cashflow
    return ticker, history, info, financials, balancesheet, cashflow, quarterly_financials, quarterly_balancesheet, quarterly_cashflow

# --- START: NEWLY ADDED/MODIFIED QUANTITATIVE FUNCTIONS ---

def calculate_mahalanobis_metrics(returns, cov_matrix, periods=252):
    """
    Calculate Mahalanobis distance and MALV for precision matrix evaluation.
    """
    try:
        cov_matrix = nearest_psd_matrix(cov_matrix)
        precision_matrix = np.linalg.inv(cov_matrix)
        mahalanobis_distances = []
        for t in range(len(returns)):
            r_t = returns.iloc[t].values.reshape(-1, 1)
            distance = np.sqrt(r_t.T @ precision_matrix @ r_t).item()
            mahalanobis_distances.append(distance)
        mahalanobis_distances = np.array(mahalanobis_distances)
        malv = np.var(mahalanobis_distances**2)
        return malv, mahalanobis_distances
    except Exception as e:
        logging.error(f"Error in Mahalanobis calculation: {e}")
        return np.nan, []

def calculate_idiosyncratic_variance(returns_df, factor_returns_df, betas):
    """
    Calculate idiosyncratic variance for each asset.
    """
    try:
        idio_vars = {}
        # Align returns and factors
        common_index = returns_df.index.intersection(factor_returns_df.index)
        returns_df_aligned = returns_df.loc[common_index]
        factor_returns_df_aligned = factor_returns_df.loc[common_index]

        for ticker in returns_df_aligned.columns:
            X = factor_returns_df_aligned.values
            y = returns_df_aligned[ticker].values

            # Simple regression to find residuals
            model = LinearRegression().fit(X, y)
            residuals = y - model.predict(X)
            idio_vars[ticker] = np.var(residuals) * 252  # Annualized

        return pd.Series(idio_vars, name='IdioVariance').fillna(0)
    except Exception as e:
        logging.error(f"Error in idiosyncratic variance calculation: {e}")
        return pd.Series(0.0, index=returns_df.columns, name='IdioVariance')

# --- FIX: THIS FUNCTION WAS RETURNING A NUMPY ARRAY INSTEAD OF A PANDAS SERIES ---
def calculate_fmp_weights(returns_df, new_factor_returns, cov_matrix, existing_factors_returns=None):
    """
    Calculate FMP weights for a new factor with optional orthogonalization against existing factors.

    Returns:
        pd.Series: A pandas Series of weights, with tickers as the index.
    """
    try:
        tickers = returns_df.columns
        cov_matrix_psd = nearest_psd_matrix(cov_matrix)
        precision_matrix = np.linalg.inv(cov_matrix_psd)

        # Align all data to a common time index
        common_idx = returns_df.index.intersection(new_factor_returns.index)
        if existing_factors_returns is not None:
            common_idx = common_idx.intersection(existing_factors_returns.index)

        aligned_returns = returns_df.loc[common_idx]
        aligned_new_factor = new_factor_returns.loc[common_idx]

        # Orthogonalize the new factor against existing factors
        if existing_factors_returns is not None and not existing_factors_returns.empty:
            aligned_existing_factors = existing_factors_returns.loc[common_idx]
            model = LinearRegression().fit(aligned_existing_factors, aligned_new_factor)
            ortho_factor_series = aligned_new_factor - pd.Series(model.predict(aligned_existing_factors), index=common_idx)
        else:
            ortho_factor_series = aligned_new_factor

        # Estimate the betas of assets to the (orthogonalized) factor
        betas = []
        for ticker in aligned_returns.columns:
            model = LinearRegression().fit(ortho_factor_series.values.reshape(-1, 1), aligned_returns[ticker].values)
            betas.append(model.coef_[0])
        B = np.array(betas).reshape(-1, 1)

        # FMP formula: w = Omega^-1 * B * (B^T * Omega^-1 * B)^-1
        Bt_Omega_B = B.T @ precision_matrix @ B
        if Bt_Omega_B[0, 0] < 1e-9: # Avoid division by zero
            return pd.Series(np.ones(len(tickers)) / len(tickers), index=tickers)

        weights = precision_matrix @ B @ np.linalg.inv(Bt_Omega_B)
        weights = weights.flatten()

        # Normalize weights
        weights = weights / np.sum(np.abs(weights))

        # --- FIX IS HERE ---
        # Return a pandas Series with tickers as the index, not a raw NumPy array
        return pd.Series(weights, index=tickers)

    except Exception as e:
        logging.error(f"Error in FMP calculation: {e}")
        # Ensure the fallback also returns a pandas Series
        return pd.Series(np.ones(len(returns_df.columns)) / len(returns_df.columns), index=returns_df.columns)

def calculate_information_metrics(forecasted_alphas_ts, realized_returns_ts):
    """
    Calculate Information Coefficient (IC) and Information Ratio (IR).
    """
    try:
        # Align the two time series
        aligned_df = pd.concat([forecasted_alphas_ts.rename('alpha'), realized_returns_ts.rename('realized')], axis=1).dropna()
        if len(aligned_df) < 20: return np.nan, np.nan

        alphas = aligned_df['alpha']
        returns = aligned_df['realized']

        # Information Coefficient (Spearman for robustness)
        ic = alphas.corr(returns, method='spearman')

        # Information Ratio (Portfolio Sharpe Ratio)
        mean_return = np.mean(returns) * 252
        volatility = np.std(returns) * np.sqrt(252)
        ir = mean_return / volatility if volatility != 0 else np.nan

        return ic, ir
    except Exception as e:
        logging.error(f"Error in information metrics: {e}")
        return np.nan, np.nan

# --- All Individual Metric Calculation Functions ---
def calculate_garch_volatility(returns, window=252, dist='t'):
    if returns.empty or len(returns) < window or returns.isna().all(): return np.nan
    try:
        scaled_returns = returns.dropna() * 100
        if len(scaled_returns) < 5 or scaled_returns.std() < 1e-6: return np.nan
        model = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist=dist)
        res = model.fit(disp='off', last_obs=None, options={'maxiter': 500})
        cond_vol = res.conditional_volatility.iloc[-1] / 100
        return cond_vol * np.sqrt(252)
    except Exception: return returns[-window:].std() * np.sqrt(252)

@lru_cache(maxsize=1024)
def calculate_returns_cached(ticker, periods_tuple):
    periods = list(periods_tuple)
    try:
        history = yf.Ticker(ticker).history(period="2y", auto_adjust=True)
        if history.empty or len(history) < max(p for p in periods if p is not None): return {f"Return_{p}d": np.nan for p in periods}
        returns = {}
        for period in periods:
            if len(history) > period:
                returns[f"Return_{period}d"] = (history['Close'].iloc[-1] / history['Close'].iloc[-(period+1)] - 1) * 100
            else:
                returns[f"Return_{period}d"] = np.nan
        return returns
    except Exception: return {f"Return_{p}d": np.nan for p in periods}

def calculate_log_log_utility(returns):
    if returns.empty or returns.isna().all(): return np.nan
    try:
        positive_returns = returns[returns > 0]
        if positive_returns.empty: return np.nan
        log_returns = np.log1p(positive_returns)
        log_log_utility = np.mean(np.log1p(log_returns))
        return log_log_utility if np.isfinite(log_log_utility) else np.nan
    except Exception: return np.nan
def simulate_historical_pure_returns(pure_returns_today):
    """
    SIMULATES a history of past Pure Factor Return tables.
    In a real system, you would load this from a database or CSV.
    """
    if pure_returns_today is None:
        return []

    historical_data = []
    for i in range(12): # Simulate 12 past "monthly" runs
        # Create a noisy, slightly different version for past months
        noise = np.random.normal(0, 0.5, len(pure_returns_today))
        drift = (12 - i) / 12 * 0.1 # Make older data slightly different
        simulated_series = pure_returns_today + noise + drift
        historical_data.append(simulated_series)

    historical_data.append(pure_returns_today) # Add today's data
    return historical_data


def analyze_coefficient_stability(historical_data):
    """
    Analyzes the stability of factor coefficients over time to find the most robust factors.
    This is the core of the new strategy.
    """
    if not historical_data:
        return pd.DataFrame()

    # Combine all historical Series into one DataFrame
    df = pd.concat(historical_data, axis=1)
    df.columns = [f'run_{i}' for i in range(len(df.columns))]

    # Calculate stability metrics for each factor
    stability_metrics = pd.DataFrame(index=df.index)
    stability_metrics['mean_coeff'] = df.mean(axis=1)
    stability_metrics['std_coeff'] = df.std(axis=1)

    # Calculate the percentage of times the coefficient was positive
    stability_metrics['pct_positive'] = (df > 0).sum(axis=1) / len(df.columns)

    # Calculate the Sharpe Ratio of the coefficient (our "Stability Score")
    # This is the signal-to-noise ratio of the factor itself.
    # Use a small epsilon to avoid division by zero for perfectly stable (but useless) factors.
    stability_metrics['sharpe_ratio_coeff'] = stability_metrics['mean_coeff'] / (stability_metrics['std_coeff'] + 1e-6)

    return stability_metrics.sort_values(by='sharpe_ratio_coeff', key=abs, ascending=False)


def set_weights_from_stability(stability_df, all_metrics, reverse_metric_map):
    """
    Sets final portfolio weights based on the factor's stability (Coefficient Sharpe Ratio).
    """
    if stability_df.empty or 'sharpe_ratio_coeff' not in stability_df.columns:
        return {metric: 0.0 for metric in all_metrics}, pd.DataFrame()

    # The weight is directly proportional to the absolute value of the stability score
    scores = stability_df['sharpe_ratio_coeff'].abs()

    # Normalize scores to sum to 100
    total_score = scores.sum()
    if total_score == 0:
        return {metric: 0.0 for metric in all_metrics}, stability_df

    final_weights = (scores / total_score) * 100

    # Build final dictionary
    final_weights_dict = {metric: 0.0 for metric in all_metrics}
    for short_name, weight in final_weights.items():
        long_name = METRIC_NAME_MAP.get(short_name, short_name)
        if long_name in final_weights_dict:
            final_weights_dict[long_name] = weight

    stability_df['Final_Weight'] = final_weights
    stability_df.fillna(0, inplace=True)

    return final_weights_dict, stability_df
def calculate_log_log_sharpe(returns, window=252, risk_free_rate=0.04):
    if returns.empty or len(returns) < window: return np.nan
    try:
        log_returns = np.log1p(returns[-window:])
        log_log_returns = np.log1p(log_returns[log_returns > 0])
        if log_log_returns.empty: return np.nan
        mean_return = log_log_returns.mean() * 252
        std_return = log_log_returns.std() * np.sqrt(252)
        daily_rf = risk_free_rate / 252
        return (mean_return - daily_rf) / std_return if std_return != 0 else np.nan
    except Exception: return np.nan

def calculate_volatility_autocorrelation(returns, window=252):
    if returns.empty or len(returns) < window: return np.nan
    try:
        squared_returns = (returns[-window:]**2).dropna()
        if len(squared_returns) < 2: return np.nan
        return squared_returns.autocorr(lag=1)
    except Exception: return np.nan

def calculate_stop_loss_impact(returns, stop_loss_level=-0.04):
    if returns.empty: return np.nan
    return (returns < stop_loss_level).mean()

def calculate_ar_coefficient(returns, lags=1, window=252):
    if returns.empty or returns.isna().all(): return np.nan
    effective_window = min(window, len(returns))
    if effective_window < 20: return np.nan
    try:
        X = returns[-effective_window:].shift(1).dropna()
        y = returns[-effective_window:].iloc[1:].reindex(X.index)
        if len(X) < 10: return np.nan
        slope, _, _, p_value, _ = linregress(X, y)
        return slope if p_value < 0.1 and np.isfinite(slope) else 0.0 # Relaxed p-value
    except Exception: return 0.0

def breakout(price, lookback=20, smooth=5):
    if len(price) < lookback: return np.nan
    roll_max = price.rolling(lookback, min_periods=lookback//2).max()
    roll_min = price.rolling(lookback, min_periods=lookback//2).min()
    roll_mean = (roll_max + roll_min) / 2.0
    output = 40.0 * ((price - roll_mean) / (roll_max - roll_min))
    smoothed_output = output.ewm(span=smooth, min_periods=smooth//2).mean()
    scaled_output = (smoothed_output + 40.0) / 80.0
    return scaled_output.iloc[-1] if not scaled_output.empty else np.nan
def calculate_volatility_adjusted_z_score(prices, period=252, ticker="Unknown", metric="Z-score", sector=None):
    """
    Calculates a robust Z-score for a price series, adjusted for volatility and sector.

    This function computes a Z-score using the median and Median Absolute Deviation (MAD)
    for robustness against outliers. It further refines the score by incorporating an
    adaptive lookback window based on the ratio of current to historical volatility,
    and applies a heuristic adjustment based on the stock's sector.

    Args:
        prices (pd.Series): A pandas Series of asset prices.
        period (int): The base period for historical calculations, default is 252.
        ticker (str): The ticker symbol for logging purposes.
        metric (str): The name of the metric being calculated for logging.
        sector (str, optional): The sector of the asset, used for applying
                                 a specific adjustment factor. Defaults to None.

    Returns:
        float: The calculated volatility-adjusted Z-score, or np.nan if calculation fails.
    """
    if prices.empty or prices.isna().any() or (prices <= 0).any():
        logging.error(f"Invalid price data for {metric} calculation (Ticker: {ticker}): empty, contains NaN, or non-positive values")
        return np.nan

    data_length = len(prices)
    min_period = 200  # Minimum required days of data
    if data_length < min_period:
        logging.error(f"Insufficient data length for {metric} calculation (Ticker: {ticker}): {data_length} < {min_period} days")
        return np.nan

    # Calculate log returns for volatility analysis
    daily_returns = np.log(prices / prices.shift(1)).dropna()

    # Calculate current and historical volatility
    current_vol = daily_returns.std() * np.sqrt(252)
    historical_vol = daily_returns.rolling(window=252).std().mean() * np.sqrt(252) if data_length >= 252 else current_vol

    # Determine volatility factor and adaptive window
    vol_factor = current_vol / historical_vol if historical_vol > 0 else 1.0
    adaptive_window = int(min(max(126, 252 * (1 + vol_factor)), 504)) # Clamp window between ~6mo and ~2yr
    use_length = min(data_length, adaptive_window)
    logging.info(f"Calculating {metric} for {ticker} with adaptive window {use_length} days (vol_factor: {vol_factor:.2f})")

    # Apply a sector-based adjustment
    sector_adjustment = 1.0
    # This check assumes 'sector_etf_map' is defined in the global scope
    if 'sector_etf_map' in globals() and sector in sector_etf_map:
        sector_adjustment = 0.9 if sector in ['Technology', 'Healthcare'] else 1.1 if sector in ['Utilities', 'Real Estate'] else 1.0

    # Calculate robust Z-score using log prices, median, and MAD
    y = np.log(prices[-use_length:]).values
    median_y = np.median(y)
    mad = np.median(np.abs(y - median_y))

    if mad == 0 or np.isnan(mad):
        logging.warning(f"Zero or NaN MAD in {metric} calculation for {ticker}")
        return np.nan

    # Scale the MAD by the current volatility regime and sector adjustment
    vol_scaling = current_vol / (historical_vol if historical_vol > 0 else 1.0)
    robust_z = 0.6745 * (y[-1] - median_y) / (mad * vol_scaling * sector_adjustment)

    return robust_z

def recalculate_relative_z_scores(top_15_df, etf_histories, period="3y", window=252, min_window=200):
    """
    Recalculates relative Z-scores for a list of stocks against their benchmark ETFs.

    This function iterates through a DataFrame of stocks, computes their relative
    price series against a specified benchmark ETF, and then uses the
    `calculate_volatility_adjusted_z_score` function to get the final score.

    Args:
        top_15_df (pd.DataFrame): DataFrame containing stock information. Must have
                                  'Ticker', 'Best_Factor', and 'Sector' columns.
        etf_histories (dict): A dictionary where keys are ETF tickers and values are
                              DataFrames of their historical price data.
        period (str): The historical data period to fetch (e.g., "3y").
        window (int): The maximum number of days for the relative calculation.
        min_window (int): The minimum number of overlapping days required.

    Returns:
        list: A list of calculated relative Z-scores, with np.nan for any failures.
    """
    relative_z_scores = []
    for idx, row in top_15_df.iterrows():
        ticker = row['Ticker']
        best_factor = row['Best_Factor']
        sector = row['Sector'] # Get sector from the dataframe
        try:
            logging.info(f"Recalculating Z-Score for {ticker}, Best_Factor: {best_factor}")

            # Fetch stock history and get pre-fetched ETF history
            history = yf.Ticker(ticker).history(period=period, auto_adjust=True, interval="1d").tz_localize(None)
            etf_history = etf_histories.get(best_factor)

            if not history.empty and etf_history is not None and not etf_history.empty:
                # Align timezones and find common date range
                etf_history = etf_history.copy()
                etf_history.index = etf_history.index.tz_localize(None)
                common_index = history.index.intersection(etf_history.index)[-window:]

                if len(common_index) >= min_window:
                    # Create aligned price series and calculate relative strength
                    aligned_stock = history['Close'][common_index]
                    aligned_etf = etf_history['Close'][common_index]
                    relative = aligned_stock / aligned_etf

                    # Validate relative series before calculating Z-score
                    if not relative.isna().any() and (relative > 0).all() and np.isfinite(relative).all():
                        z_score = calculate_volatility_adjusted_z_score(relative, period=len(common_index), ticker=ticker, sector=sector)
                        relative_z_scores.append(z_score)
                        logging.info(f"Calculated Z-Score for {ticker} vs {best_factor}: {z_score:.4f}")
                    else:
                        relative_z_scores.append(np.nan)
                        logging.warning(f"Invalid relative data for {ticker} vs {best_factor}")
                else:
                    relative_z_scores.append(np.nan)
                    logging.warning(f"Insufficient data for {ticker} vs {best_factor}: {len(common_index)} days")
            else:
                relative_z_scores.append(np.nan)
                logging.warning(f"Empty history for {ticker} or {best_factor}")
        except Exception as e:
            relative_z_scores.append(np.nan)
            logging.error(f"Error recalculating Z-Score for {ticker}: {str(e)}")

    return relative_z_scores
def calculate_piotroski_f_score(financials, balancesheet, cashflow, total_assets, roa, net_income):
    score = 0
    try:
        if roa is not None and roa > 0: score += 1
        op_cash_flow = get_value(cashflow, ['Operating Cash Flow'])
        if op_cash_flow is not None and op_cash_flow > 0: score += 1
        if op_cash_flow is not None and net_income is not None and op_cash_flow > net_income: score += 1
        long_term_debt_curr = get_value(balancesheet, ['Long Term Debt'])
        long_term_debt_prev = get_value(balancesheet, ['Long Term Debt'], 1)
        if long_term_debt_curr is not None and long_term_debt_prev is not None and long_term_debt_curr <= long_term_debt_prev: score += 1
        current_ratio_curr = get_value(balancesheet, ['Total Current Assets']) / get_value(balancesheet, ['Total Current Liabilities']) if get_value(balancesheet, ['Total Current Liabilities']) else np.nan
        current_ratio_prev = get_value(balancesheet, ['Total Current Assets'], 1) / get_value(balancesheet, ['Total Current Liabilities'], 1) if get_value(balancesheet, ['Total Current Liabilities'], 1) else np.nan
        if current_ratio_curr is not None and current_ratio_prev is not None and current_ratio_curr > current_ratio_prev: score += 1
        gross_margin_curr = get_value(financials, ['Gross Profit']) / get_value(financials, ['Total Revenue']) if get_value(financials, ['Total Revenue']) else np.nan
        gross_margin_prev = get_value(financials, ['Gross Profit'], 1) / get_value(financials, ['Total Revenue'], 1) if get_value(financials, ['Total Revenue'], 1) else np.nan
        if gross_margin_curr is not None and gross_margin_prev is not None and gross_margin_curr > gross_margin_prev: score += 1
        asset_turnover_curr = get_value(financials, ['Total Revenue']) / total_assets if total_assets else np.nan
        asset_turnover_prev = get_value(financials, ['Total Revenue'], 1) / get_value(balancesheet, ['Total Assets'], 1) if get_value(balancesheet, ['Total Assets'], 1) else np.nan
        if asset_turnover_curr is not None and asset_turnover_prev is not None and asset_turnover_curr > asset_turnover_prev: score += 1
    except Exception: pass
    return score

def calculate_lo_modified_variance(sub_series, q):
    n = len(sub_series)
    if n < 2: return np.nan
    mean_val = np.mean(sub_series)
    if np.allclose(sub_series, mean_val): return 0.0
    sample_var = np.sum((sub_series - mean_val)**2) / n
    if sample_var < 0: return np.nan
    if q <= 0 or q >= n: return sample_var
    try:
        autocovariances = smt.acovf(sub_series, adjusted=False, fft=True, nlag=q)[1:]
        if len(autocovariances) != q: return sample_var
    except Exception: return sample_var
    autocovariance_sum = 0.0
    for j in range(q):
        weight = 1.0 - ((j + 1) / (q + 1.0))
        autocovariance_sum += weight * autocovariances[j]
    modified_var = sample_var + 2.0 * autocovariance_sum
    return max(0.0, modified_var)

def calculate_hurst_lo_modified(series, min_n=10, max_n=None, q_method='auto'):
    if isinstance(series, pd.Series): series = series.values
    series = series[~np.isnan(series)]
    N = len(series)
    if max_n is None: max_n = N // 2
    max_n = min(max_n, N - 1)
    min_n = max(2, min_n)
    if N < 20 or min_n >= max_n: return np.nan, pd.DataFrame()
    ns = np.unique(np.geomspace(min_n, max_n, num=20, dtype=int))
    ns = [n_val for n_val in ns if n_val >= min_n]
    if not ns: return np.nan, pd.DataFrame()
    rs_values, valid_ns = [], []
    for n in ns:
        q = 0
        if isinstance(q_method, int): q = max(0, min(q_method, n - 1))
        elif q_method == 'auto' and n > 10: q = max(0, min(int(np.floor(1.1447 * (n**(1/3)))), n - 1))
        rs_chunk = []
        num_chunks = N // n
        if num_chunks == 0: continue
        for i in range(num_chunks):
            chunk = series[i*n : (i+1)*n]
            mean = np.mean(chunk)
            if np.allclose(chunk, mean, rtol=1e-8, atol=1e-10): continue
            mean_adjusted = chunk - mean
            cum_dev = np.cumsum(mean_adjusted)
            R = np.ptp(cum_dev)
            if pd.isna(R) or R < 0: continue
            modified_var = calculate_lo_modified_variance(chunk, q)
            if pd.isna(modified_var) or modified_var < 1e-12: continue
            S_q = np.sqrt(modified_var)
            rs = R / S_q
            if not pd.isna(rs) and rs >= 0: rs_chunk.append(rs)
        if rs_chunk:
            rs_values.append(np.mean(rs_chunk))
            valid_ns.append(n)
    if len(valid_ns) < 3: return np.nan, pd.DataFrame()
    results_df = pd.DataFrame({'interval': valid_ns, 'rs_mean': rs_values}).dropna()
    if len(results_df) < 3: return np.nan, pd.DataFrame()
    try:
        hurst, _, _, _, _ = linregress(np.log(results_df['interval']), np.log(results_df['rs_mean']))
        return hurst, results_df
    except Exception: return np.nan, pd.DataFrame()

# --- FIX: THIS IS THE COMPLETE AND CORRECTED FUNCTION. REPLACE THE EXISTING ONE. ---
def process_single_ticker(ticker_symbol, etf_histories, sector_etf_map):
    try:
        _, history, info, financials, balancesheet, cashflow, _, _, _ = fetch_ticker_data(ticker_symbol)

        if history.empty or not info:
            failed_data = {col: np.nan for col in columns}
            failed_data['Ticker'] = ticker_symbol
            failed_data['Name'] = f"{ticker_symbol} (Failed to fetch)"
            return [failed_data.get(col) for col in columns], pd.Series()

        # 1. Initialize a dictionary with all column names
        data = {col: np.nan for col in columns}

        # 2. Populate the dictionary by key, ensuring no misalignment
        data['Ticker'] = ticker_symbol
        data['Name'] = info.get('longName', 'N/A')
        data['Sector'] = info.get('sector', 'Unknown')

        # --- Basic Info & Financial Data Extraction ---
        data['Market_Cap'] = info.get('marketCap')
        data['Dividend_Yield'] = info.get('dividendYield', 0) * 100
        data['PE_Ratio'] = info.get('trailingPE')
        data['EPS_Diluted'] = info.get('trailingEps')
        shares_outstanding = info.get('sharesOutstanding')
        current_price = info.get('currentPrice', info.get('regularMarketPrice'))
        data['Insider_Ownership_Ratio'] = info.get('heldPercentInsiders', 0) * 100
        data['Institutional_Ownership_Ratio'] = info.get('heldPercentInstitutions', 0) * 100
        data['Audit_Risk'], data['Board_Risk'], data['Compensation_Risk'], data['Shareholder_Rights_Risk'], data['Overall_Risk'] = info.get('auditRisk'), info.get('boardRisk'), info.get('compensationRisk'), info.get('shareHolderRightsRisk'), info.get('overallRisk')

        revenue = get_value(financials, ['Total Revenue', 'TotalRevenue'])
        gross_profit = get_value(financials, ['Gross Profit', 'GrossProfit'])
        net_income = get_value(financials, ['Net Income', 'NetIncome'])
        operating_income = get_value(financials, ['Operating Income', 'OperatingIncome'])
        interest_expense = get_value(financials, ['Interest Expense', 'InterestExpense'])
        ebit = get_value(financials, ['Ebit', 'EBIT'])
        total_assets = get_value(balancesheet, ['Total Assets', 'TotalAssets'])
        total_liabilities = get_value(balancesheet, ['Total Liabilities', 'TotalLiab'])
        intangibles = (get_value(balancesheet, ['Intangible Assets', 'IntangibleAssets']) or 0) + (get_value(balancesheet, ['Goodwill']) or 0)
        total_equity = get_value(balancesheet, ['Total Stockholder Equity', 'TotalStockholderEquity'])
        current_assets = get_value(balancesheet, ['Total Current Assets', 'TotalCurrentAssets'])
        current_liabilities = get_value(balancesheet, ['Total Current Liabilities', 'TotalCurrentLiabilities'])
        inventory = get_value(balancesheet, ['Inventory']) or 0
        cogs = get_value(financials, ['Cost Of Revenue', 'CostOfRevenue'])
        operating_cash_flow = get_value(cashflow, ['Operating Cash Flow', 'TotalCashFromOperatingActivities'])
        capex = get_value(cashflow, ['Capital Expenditure', 'CapitalExpenditures'])
        depreciation = get_value(cashflow, ['Depreciation And Amortization', 'Depreciation'])
        dividends_paid = get_value(cashflow, ['Dividends Paid', 'DividendsPaid']) or 0
        buybacks = get_value(cashflow, ['Repurchase Of Capital Stock', 'RepurchaseOfStock']) or 0
        fcf = (operating_cash_flow if pd.notna(operating_cash_flow) else 0) + (capex if pd.notna(capex) else 0)

        # --- Ratio Calculations ---
        data['Current_Ratio'] = current_assets / current_liabilities if current_liabilities and current_liabilities > 0 else np.nan
        data['Quick_Ratio'] = (current_assets - inventory) / current_liabilities if current_liabilities and current_liabilities > 0 else np.nan
        data['Debt_Ratio'] = total_liabilities / total_assets if total_assets and total_assets > 0 else np.nan
        data['Liabilities_to_Equity'] = total_liabilities / total_equity if total_equity and total_equity > 0 else np.nan
        data['Gross_Profit_Margin'] = (gross_profit / revenue) * 100 if revenue and revenue > 0 else np.nan
        data['Operating_Margin'] = (operating_income / revenue) * 100 if revenue and revenue > 0 else np.nan
        data['Net_Profit_Margin'] = (net_income / revenue) * 100 if revenue and revenue > 0 else np.nan
        data['ROA'] = (net_income / total_assets) * 100 if total_assets and total_assets > 0 else np.nan
        data['ROE'] = (net_income / total_equity) * 100 if total_equity and total_equity > 0 else np.nan
        data['PS_Ratio'] = data['Market_Cap'] / revenue if revenue and data['Market_Cap'] and revenue > 0 else np.nan
        data['FCF_Yield'] = (fcf / data['Market_Cap']) * 100 if data['Market_Cap'] and data['Market_Cap'] > 0 else np.nan
        data['Sales_Per_Share'] = revenue / shares_outstanding if shares_outstanding and shares_outstanding > 0 else np.nan
        data['FCF_Per_Share'] = fcf / shares_outstanding if shares_outstanding and shares_outstanding > 0 else np.nan
        data['Asset_Turnover'] = revenue / total_assets if total_assets and total_assets > 0 else np.nan
        data['CapEx_to_DepAmor'] = abs(capex or 0) / depreciation if depreciation and depreciation > 0 else np.nan
        data['Dividends_to_FCF'] = abs(dividends_paid) / fcf if fcf and fcf > 0 else np.nan
        data['Interest_Coverage'] = ebit / abs(interest_expense or 1) if interest_expense is not None else np.nan
        data['Inventory_Turnover'] = cogs / inventory if inventory and inventory > 0 else np.nan
        data['Share_Buyback_to_FCF'] = abs(buybacks) / fcf if fcf and fcf > 0 else np.nan
        data['Dividends_Plus_Buyback_to_FCF'] = (abs(dividends_paid) + abs(buybacks)) / fcf if fcf and fcf > 0 else np.nan
        data['Earnings_Yield'] = (data['EPS_Diluted'] / current_price) * 100 if current_price and current_price > 0 else np.nan
        data['FCF_to_Net_Income'] = fcf / net_income if net_income and net_income > 0 else np.nan
        data['Tangible_Book_Value'] = total_assets - intangibles - total_liabilities if all(pd.notna([total_assets, intangibles, total_liabilities])) else np.nan
        data['Return_On_Tangible_Equity'] = (net_income / data['Tangible_Book_Value']) * 100 if pd.notna(data['Tangible_Book_Value']) and data['Tangible_Book_Value'] != 0 else np.nan
        data['Earnings_Growth_Rate_5y'] = info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else np.nan
        data['Revenue_Growth_Rate_5y'] = info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else np.nan
        data['Piotroski_F-Score'] = calculate_piotroski_f_score(financials, balancesheet, cashflow, total_assets, data['ROA'], net_income)
        nopat = operating_income * (1 - 0.25) if operating_income else np.nan
        invested_capital = total_assets - current_liabilities if all(pd.notna([total_assets, current_liabilities])) else np.nan
        data['ROIC'] = (nopat / invested_capital) * 100 if invested_capital and invested_capital > 0 else np.nan
        data['Cash_ROIC'] = (fcf / invested_capital) * 100 if invested_capital and invested_capital > 0 else np.nan

        # --- Time-series, Technicals, and Factor Calculations ---
        log_returns = pd.Series()
        if not history.empty and 'Close' in history.columns:
            log_returns = np.log(history['Close'] / history['Close'].shift(1)).dropna()
            if not log_returns.empty:
                data['GARCH_Vol'] = calculate_garch_volatility(log_returns)
                data['AR_Coeff'] = calculate_ar_coefficient(log_returns)
                data['Log_Log_Utility'] = calculate_log_log_utility(log_returns)
                data['Log_Log_Sharpe'] = calculate_log_log_sharpe(log_returns)
                data['Vol_Autocorr'] = calculate_volatility_autocorrelation(log_returns)
                data['Stop_Loss_Impact'] = calculate_stop_loss_impact(log_returns)
                hurst, _ = calculate_hurst_lo_modified(log_returns)
                data['Hurst_Exponent'] = hurst
                data['Trend'] = breakout(history['Close'])
                data['Dollar_Volume_90D'] = (history['Volume'] * history['Close']).rolling(90).mean().iloc[-1]
                data['Momentum'] = (history['Close'].iloc[-1] / history['Close'].iloc[-252] - 1) * 100 if len(history) > 252 else np.nan

                spy_hist = etf_histories.get('SPY')
                if spy_hist is not None and not spy_hist.empty:
                    spy_returns = np.log(spy_hist['Close'] / spy_hist['Close'].shift(1)).dropna()
                    common_idx = log_returns.index.intersection(spy_returns.index)
                    if len(common_idx) > 30:
                        slope, _, _, _, _ = linregress(spy_returns[common_idx], log_returns[common_idx])
                        data['Beta_to_SPY'] = slope

                # --- THIS IS THE CRITICAL LOGIC BLOCK THAT WAS RESTORED ---
                rolling_correlations = {}
                for etf, etf_history in etf_histories.items():
                    if etf_history is not None and not etf_history.empty:
                        etf_returns = np.log(etf_history['Close'] / etf_history['Close'].shift(1)).dropna()
                        common_idx = log_returns.index.intersection(etf_returns.index)
                        if len(common_idx) > 90:
                            corr = log_returns.loc[common_idx].corr(etf_returns.loc[common_idx])
                            if pd.notna(corr): rolling_correlations[etf] = corr

                if rolling_correlations:
                    best_factor_ticker = max(rolling_correlations, key=lambda k: abs(rolling_correlations.get(k, 0)))
                    data['Best_Factor'] = best_factor_ticker
                    data['Correlation_Score'] = rolling_correlations.get(best_factor_ticker)

                    best_etf_hist = etf_histories.get(best_factor_ticker)
                    if best_etf_hist is not None:
                        common_idx_z = history.index.intersection(best_etf_hist.index)
                        if len(common_idx_z) > 200:
                            relative_strength = history['Close'][common_idx_z] / best_etf_hist['Close'][common_idx_z]
                            data['Relative_Z_Score'] = calculate_volatility_adjusted_z_score(relative_strength, ticker=ticker_symbol, sector=data['Sector'])

        returns_perf = calculate_returns_cached(ticker_symbol, tuple([21, 63, 126, 252]))
        data.update({f"Return_{p}d": returns_perf.get(f"Return_{p}d") for p in [21, 63, 126, 252]})
        data['Growth'] = data.get('Sales_Growth_YOY')

        # --- RESTORED LOGIC for other scores ---
        data['Q_Score'] = min(data.get('Quick_Ratio', 0) / 5.0, 1.0) if pd.notna(data.get('Quick_Ratio')) and data.get('Quick_Ratio') > 0 else 0.0
        data['Coverage_Score'] = min(data.get('Interest_Coverage', 0) / 10.0, 1.0) if pd.notna(data.get('Interest_Coverage')) and data.get('Interest_Coverage') > 0 else 0.0

        vision_score = 0
        if data.get('Sector') == 'Technology' and pd.notna(net_income) and net_income < 0: vision_score += 5
        if pd.notna(data.get('Sales_Growth_TTM')) and data.get('Sales_Growth_TTM') > 20: vision_score += 3
        data['Vision'] = min(vision_score / 8.0, 1.0)

        pe = data.get('PE_Ratio'); ps = data.get('PS_Ratio')
        if pd.notna(pe) and pd.notna(ps) and (pe + ps) > 0:
            data['Value_Factor'] = min(1 / ((pe + ps) / 2.0), 1.0)
        else:
            data['Value_Factor'] = 0.0

        profit_metrics = [data.get('ROE'), data.get('ROIC'), data.get('Net_Profit_Margin')]
        valid_profit_metrics = [m for m in profit_metrics if pd.notna(m)]
        data['Profitability_Factor'] = min(np.mean(valid_profit_metrics) / 100.0, 1.0) if valid_profit_metrics else 0.0

        # --- END OF RESTORED LOGIC ---

        # 3. Build the final list in the correct order using the master `columns` list
        result_list = [data.get(col) for col in columns]
        return result_list, log_returns

    except Exception as e:
        logging.error(f"Critical error processing {ticker_symbol}: {e}", exc_info=True)
        failed_data = {col: np.nan for col in columns}
        failed_data['Ticker'] = ticker_symbol
        failed_data['Name'] = f"{ticker_symbol} (Processing Error)"
        return [failed_data.get(col) for col in columns], pd.Series()

# --- FIX: Replaced this entire function to fix the 'inplace' warning and improve cleaning ---
@st.cache_data
def process_tickers(_tickers, _etf_histories, _sector_etf_map):
    results, returns_dict, failed_tickers = [], {}, []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(process_single_ticker, ticker, _etf_histories, _sector_etf_map): ticker for ticker in _tickers}
        for future in tqdm(as_completed(future_to_ticker), total=len(_tickers), desc="Processing All Ticker Metrics"):
            ticker = future_to_ticker[future]
            try:
                result, returns = future.result()
                if result and pd.notna(result[1]):  # Check if Name is not NaN
                    results.append(result)
                    if returns is not None and not returns.empty:
                        returns_dict[ticker] = returns
                else:
                    failed_tickers.append(ticker)
            except Exception as e:
                logging.error(f"Failed to process {ticker} in future: {e}")
                failed_tickers.append(ticker)

    if not results:
        return pd.DataFrame(columns=columns), failed_tickers, {}

    results_df = pd.DataFrame(results, columns=columns)

    numeric_cols = [c for c in columns if c not in ['Ticker', 'Name', 'Sector', 'Best_Factor', 'Risk_Flag']]
    results_df[numeric_cols] = results_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Corrected data cleaning loop to avoid 'inplace' on a copy
    for col in results_df.select_dtypes(include=np.number).columns:
        if results_df[col].isna().all():
            results_df[col] = 0.0
        else:
            median_val = results_df[col].median()
            # The correct way: reassign the column
            results_df[col] = results_df[col].fillna(median_val)

        if results_df[col].var() < 1e-8:
            results_df[col] += np.random.normal(0, 0.01, len(results_df))

    return results_df.infer_objects(copy=False), failed_tickers, returns_dict

# --- FIX: REPLACED ENTIRE FUNCTION TO BE MORE ROBUST AND PREVENT LINALGWARNING ---


def check_multicollinearity(X, characteristics, vif_threshold=5.0):
    """
    Iteratively removes features with high Variance Inflation Factor (VIF)
    to handle multicollinearity in a robust way, suppressing expected warnings.

    This function first cleans the data by removing zero-variance columns and
    handling non-finite values. It then enters a loop that calculates VIF for
    all features, identifies the feature with the highest VIF, and removes it
    if it exceeds the threshold. This process repeats until all remaining
    features have a VIF below the threshold. The specific `RuntimeWarning` for
    division by zero (caused by perfect multicollinearity) is suppressed.

    Args:
        X (pd.DataFrame): The input feature matrix.
        characteristics (list): A list of column names in X to check for VIF.
        vif_threshold (float): The threshold above which features are removed.

    Returns:
        list: A list of feature names with VIF below the threshold.
    """
    # 1. Handle edge cases where VIF is not applicable
    if X.empty or X.shape[1] < 2:
        return characteristics

    # 2. Pre-process the data for VIF calculation
    # Work on a copy of the relevant columns
    X_vif = X[characteristics].copy()
    
    # Replace inf/-inf with NaN and then fill NaNs with the column median
    X_vif.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_vif.fillna(X_vif.median(), inplace=True)

    # Drop columns with near-zero variance, as they are constant and cause VIF to fail
    variances = X_vif.var()
    non_zero_var_cols = variances[variances > 1e-8].index.tolist()
    X_vif = X_vif[non_zero_var_cols]

    if X_vif.shape[1] < 2:
        return X_vif.columns.tolist()

    # 3. Iteratively remove features with the highest VIF
    while True:
        # Create a DataFrame to hold VIF values
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_vif.columns

        # Suppress the specific RuntimeWarning during VIF calculation
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            try:
                vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
            except Exception:
                # In case of a different error (e.g., LinAlgError), drop the last column as a safe fallback
                if not X_vif.empty:
                    X_vif = X_vif.iloc[:, :-1]
                    continue
                else:
                    break

        # Find the feature with the maximum VIF
        max_vif = vif_data['VIF'].max()

        # If the highest VIF is below the threshold, the process is done
        if max_vif < vif_threshold:
            break

        # Otherwise, find the feature with the highest VIF and remove it
        feature_to_drop = vif_data.sort_values('VIF', ascending=False)['feature'].iloc[0]
        X_vif = X_vif.drop(columns=[feature_to_drop])
        
        # Stop if we run out of features
        if X_vif.shape[1] < 2:
            break
            
    # 4. Return the list of features that passed the VIF check
    final_characteristics = X_vif.columns.tolist()
    return final_characteristics

# --- FIX: Replaced this entire function to correct inefficiency and bugs ---
def calculate_portfolio_factor_correlations(weighted_df, etf_histories, period="3y", min_days=240):
    """
    Calculates the correlation of a weighted portfolio's returns against a list of ETF returns.
    This version is robust and efficient.
    """
    logging.info("Starting robust portfolio factor correlations calculation")
    correlations = pd.Series(dtype=float)

    # Step 1: Compute the weighted portfolio returns series once
    portfolio_returns = None
    if 'Weight' not in weighted_df.columns or weighted_df.empty:
        logging.warning("Weighted DataFrame is empty or missing 'Weight' column.")
        return correlations

    for idx, row in weighted_df.iterrows():
        ticker = row['Ticker']
        weight = row['Weight']
        try:
            history = yf.Ticker(ticker).history(period=period, auto_adjust=True, interval="1d").tz_localize(None)
            if history.empty or 'Close' not in history.columns:
                continue

            # Calculate returns for each asset in the portfolio
            returns = history['Close'].pct_change(fill_method=None).dropna()

            if returns.empty:
                continue

            if portfolio_returns is None:
                portfolio_returns = returns * weight
            else:
                portfolio_returns = portfolio_returns.add(returns * weight, fill_value=0)
        except Exception as e:
            logging.error(f"Error fetching history for {ticker} in factor correlation: {e}")
            continue

    if portfolio_returns is None or portfolio_returns.empty:
        logging.error("Failed to compute portfolio returns for factor correlation.")
        return correlations # Return an empty series

    portfolio_returns = portfolio_returns.dropna()

    # Step 2: Compute correlations with each ETF
    for etf, etf_history in etf_histories.items():
        if etf_history.empty or 'Close' not in etf_history.columns:
            continue
        try:
            # Calculate returns ONLY for the current ETF in the loop
            etf_returns = etf_history['Close'].pct_change(fill_method=None).dropna()

            if etf_returns.empty:
                continue

            # Find common dates between the pre-calculated portfolio and the current ETF
            common_index = portfolio_returns.index.intersection(etf_returns.index)
            if len(common_index) < min_days:
                continue

            # Align the two series to the common index for correlation
            aligned_portfolio = portfolio_returns.loc[common_index]
            aligned_etf = etf_returns.loc[common_index]

            # Calculate and store the correlation
            corr = aligned_portfolio.corr(aligned_etf)
            if np.isfinite(corr):
                correlations[etf] = corr

        except Exception as e:
            logging.error(f"Error calculating correlation for ETF {etf}: {e}")
            continue

    # Fallback to SPY if no other correlations could be calculated
    if correlations.empty and portfolio_returns is not None and not portfolio_returns.empty:
        logging.warning("No valid correlations computed, falling back to SPY")
        spy_history = etf_histories.get('SPY')
        if spy_history is not None and not spy_history.empty:
            spy_returns = spy_history['Close'].pct_change(fill_method=None).dropna()
            common_index = portfolio_returns.index.intersection(spy_returns.index)
            if len(common_index) >= min_days:
                corr = portfolio_returns.loc[common_index].corr(spy_returns.loc[common_index])
                correlations['SPY'] = corr if np.isfinite(corr) else 0.5
            else:
                correlations['SPY'] = 0.5

    return correlations.sort_values(ascending=False)

# --- FIX: REPLACED ENTIRE FUNCTION TO USE PCA AND AVOID LINALGWARNING ---
def calculate_pure_returns(df, characteristics, target='Return_252d', vif_threshold=5, use_pca=True, pca_variance_threshold=0.95):
    """
    Calculates pure factor returns using a robust cross-sectional regression.
    This version uses PCA to handle severe multicollinearity, preventing LinAlgWarning.
    """
    if df.empty or target not in df.columns or df[target].isnull().all():
        return pd.Series(dtype=float, name="PureReturns")

    y = pd.to_numeric(df[target], errors='coerce')
    valid_characteristics = [col for col in characteristics if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    X = df[valid_characteristics].copy().replace([np.inf, -np.inf], np.nan)

    valid_indices = y.dropna().index
    X, y = X.loc[valid_indices], y.loc[valid_indices]
    if X.empty or y.empty or len(y) < 20:
        logging.warning(f"Insufficient data for pure returns calculation: {len(y)} samples.")
        return pd.Series(dtype=float, name="PureReturns")

    for col in X.columns:
        if X[col].isna().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
        if X[col].min() >= 0 and X[col].quantile(0.75) > 1000: X[col] = np.log1p(X[col])
        if X[col].var() < 1e-8: X[col] += np.random.normal(0, 1e-4, len(X))

    # Use a more aggressive VIF check first
    final_characteristics = check_multicollinearity(X, valid_characteristics, vif_threshold)
    if not final_characteristics:
        logging.warning("No valid characteristics left after VIF check.")
        return pd.Series(dtype=float, name="PureReturns")

    X = X[final_characteristics]

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        model = Ridge(alpha=1.0, solver='auto') # Use a robust solver

        if use_pca:
            # --- PCA IMPLEMENTATION ---
            # Keep enough components to explain 95% of the variance
            pca = PCA(n_components=pca_variance_threshold)
            X_pca = pca.fit_transform(X_scaled)

            # Regress on the uncorrelated principal components
            model.fit(X_pca, y)

            # Transform coefficients back to the original feature space for interpretation
            # This is the key step: model.coef_ are for PCA space, we need them for original features
            original_space_coefs = pca.inverse_transform(model.coef_.reshape(1, -1))[0]

        else:
            # Fallback to the old method if PCA is turned off
            model.fit(X_scaled, y)
            original_space_coefs = model.coef_

        # Unscale coefficients to be interpretable
        scaler_scale = scaler.scale_
        scaler_scale[scaler_scale < 1e-8] = 1e-8
        unscaled_coefs = original_space_coefs / scaler_scale

        return pd.Series(unscaled_coefs, index=final_characteristics, name="PureReturns").clip(lower=-10.0, upper=10.0)

    except Exception as e:
        logging.error(f"Pure returns regression failed: {e}")
        return pd.Series(dtype=float, name="PureReturns")

# --- This is the NEW function to add/replace the old one ---
def aggregate_stability_and_set_weights(stability_results, all_metrics, reverse_metric_map):
    """
    Aggregates stability metrics from multiple time horizons and sets final portfolio weights.
    Weights are based on a combination of average signal strength (sharpe ratio) and consistency across horizons.
    """
    if not stability_results:
        return {metric: 0.0 for metric in all_metrics}, pd.DataFrame()

    # Consolidate all factors from all horizons
    all_factors = set()
    for horizon, df in stability_results.items():
        all_factors.update(df.index)

    # Create the aggregated rationale DataFrame
    agg_df = pd.DataFrame(index=list(all_factors))
    agg_df['avg_sharpe_coeff'] = 0.0
    agg_df['consistency_score'] = 0.0 # How often the sign matches the average sign
    agg_df['horizons_present'] = 0

    # Aggregate metrics for each factor
    for factor in agg_df.index:
        sharpes = []
        for horizon, df in stability_results.items():
            if factor in df.index:
                sharpes.append(df.loc[factor, 'sharpe_ratio_coeff'])

        if not sharpes:
            continue

        avg_sharpe = np.mean(sharpes)
        agg_df.loc[factor, 'avg_sharpe_coeff'] = avg_sharpe
        agg_df.loc[factor, 'horizons_present'] = len(sharpes)

        # Calculate consistency: % of times the sharpe had the same sign as the average
        if avg_sharpe != 0:
            sign_of_avg = np.sign(avg_sharpe)
            same_sign_count = sum(1 for s in sharpes if np.sign(s) == sign_of_avg)
            agg_df.loc[factor, 'consistency_score'] = same_sign_count / len(sharpes)
        else:
            agg_df.loc[factor, 'consistency_score'] = 0.0

    # Calculate final score: reward both magnitude and consistency
    # We use a power on consistency to heavily reward factors that work across all horizons
    agg_df['Final_Score'] = agg_df['avg_sharpe_coeff'].abs() * (agg_df['consistency_score'] ** 2)
    agg_df = agg_df.sort_values('Final_Score', ascending=False).fillna(0)

    # Normalize scores to get final weights
    total_score = agg_df['Final_Score'].sum()
    if total_score > 0:
        agg_df['Final_Weight'] = (agg_df['Final_Score'] / total_score) * 100
    else:
        agg_df['Final_Weight'] = 0.0

    # Build the final dictionary for all possible metrics
    final_weights_dict = {metric: 0.0 for metric in all_metrics}
    for short_name, row in agg_df.iterrows():
        long_name = METRIC_NAME_MAP.get(short_name, short_name)
        if long_name in final_weights_dict:
            final_weights_dict[long_name] = row['Final_Weight']

    return final_weights_dict, agg_df

# --- FIX: QUANTITATIVE ENHANCEMENT - USE LEDOIT-WOLF AND FIX `inplace` ---
def calculate_correlation_matrix(tickers, returns_dict, window=90):
    """
    Calculates a robust, positive semi-definite correlation and covariance matrix.

    This function uses the Ledoit-Wolf shrinkage estimator for covariance, which is
    well-suited for financial data (many assets, fewer time periods). It then
    derives the correlation matrix and ensures it is positive semi-definite (PSD).

    Args:
        tickers (list): The complete list of tickers for the final matrix shape.
        returns_dict (dict): A dictionary where keys are tickers and values are pd.Series of returns.
        window (int): The number of recent trading days to use for the calculation.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The final, PSD-corrected correlation matrix.
            - pd.DataFrame: The annualized covariance matrix.
    """
    n = len(tickers)
    if n == 0 or not returns_dict:
        # Return empty dataframes if there's nothing to process
        return pd.DataFrame(), pd.DataFrame()

    # Create a DataFrame from the pre-calculated returns
    returns_df = pd.DataFrame(returns_dict).reindex(columns=tickers)

    # Take the recent window of returns
    aligned_returns = returns_df.tail(window)

    # --- FIX: Replaced 'inplace=True' with direct reassignment for safety and clarity ---
    # Drop columns (stocks) that have no data at all in the window
    aligned_returns = aligned_returns.dropna(axis=1, how='all')
    # Fill any remaining gaps (e.g., a single missing day) with 0.0
    aligned_returns = aligned_returns.fillna(0.0)

    valid_tickers = aligned_returns.columns.tolist()
    if len(valid_tickers) < 2:
        # Not enough data to compute a matrix, return identity matrices
        identity = pd.DataFrame(np.eye(n), index=tickers, columns=tickers)
        return identity, identity

    try:
        # Use Ledoit-Wolf shrinkage to get a well-conditioned covariance matrix
        lw = LedoitWolf()
        lw.fit(aligned_returns)

        # Annualize the covariance matrix (daily variance * 252)
        cov_matrix_values = lw.covariance_ * 252

        # Build the full-sized covariance matrix, handling tickers that were dropped
        cov_matrix_full = pd.DataFrame(np.eye(n) * np.mean(np.diag(cov_matrix_values)), index=tickers, columns=tickers)
        cov_matrix_full.loc[valid_tickers, valid_tickers] = cov_matrix_values

        # Derive the correlation matrix from the covariance matrix
        vols = np.sqrt(np.diag(cov_matrix_values))
        vols[vols < 1e-8] = 1.0 # Avoid division by zero for non-volatile assets
        corr_matrix_values = cov_matrix_values / np.outer(vols, vols)

        # Build the full-sized correlation matrix
        corr_matrix_full = pd.DataFrame(np.eye(n), index=tickers, columns=tickers)
        corr_matrix_full.loc[valid_tickers, valid_tickers] = corr_matrix_values

        # Ensure the final correlation matrix is positive semi-definite (PSD) for stability
        final_corr = pd.DataFrame(nearest_psd_matrix(corr_matrix_full.values), index=tickers, columns=tickers)

    except Exception as e:
        logging.error(f"Ledoit-Wolf estimation failed: {e}. Falling back to identity matrix.")
        identity = pd.DataFrame(np.eye(n), index=tickers, columns=tickers)
        return identity, identity

    return final_corr, cov_matrix_full

# --- FIX: QUANTITATIVE ENHANCEMENT - ADD DIVERSIFICATION CONSTRAINT ---
def calculate_weights(returns_df, method="equal", cov_matrix=None, factor_returns=None, betas=None):
    """
    Calculate portfolio weights with various methods, including FMP and Alpha-Orthogonal.
    """
    n_assets = len(returns_df.columns)
    tickers = returns_df.columns

    try:
        if method == "equal":
            return pd.Series(np.ones(n_assets) / n_assets, index=tickers)

        elif method == "inv_vol":
            vols = returns_df.std() * np.sqrt(252)
            inv_vols = 1 / vols.replace(0, 1e-6)
            return inv_vols / inv_vols.sum()

        elif method == "log_log_sharpe":
            mu = returns_df.mean() * 252
            if cov_matrix is None:
                cov_matrix = returns_df.cov() * 252
            cov_matrix = nearest_psd_matrix(cov_matrix)
            w = cp.Variable(n_assets)

            # --- FIX: Add risk aversion and weight constraints ---
            gamma = cp.Parameter(nonneg=True, value=0.5) # Risk aversion parameter
            max_weight = 0.15 # Max 15% in any single stock

            objective = cp.Maximize(mu.values @ w - gamma * cp.quad_form(w, cov_matrix))
            constraints = [
                cp.sum(w) == 1,
                w >= 0,
                w <= max_weight # Diversification constraint
            ]
            # --- END FIX ---

            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.SCS)

            if w.value is not None and w.value.sum() > 1e-6:
                 return pd.Series(w.value / np.sum(w.value), index=tickers)
            else:
                 return pd.Series(np.ones(n_assets) / n_assets, index=tickers)


        elif method == "fmp":
            if factor_returns is None or cov_matrix is None:
                raise ValueError("Factor returns and covariance matrix required for FMP.")
            weights = calculate_fmp_weights(returns_df, factor_returns, cov_matrix)
            return pd.Series(weights, index=tickers)

        elif method == "alpha_orthogonal":
            if betas is None or returns_df is None:
                raise ValueError("Betas and returns required for alpha_orthogonal.")

            alpha = returns_df.mean() * 252
            w = cp.Variable(n_assets)
            objective = cp.Maximize(alpha.values @ w)
            constraints = [cp.sum(w) == 1, w >= 0, betas.values.T @ w == 0]
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.SCS)

            if w.value is not None and w.value.sum() > 1e-6:
                return pd.Series(w.value / np.sum(w.value), index=tickers)
            else:
                return pd.Series(np.ones(n_assets) / n_assets, index=tickers)

        else:
            raise ValueError(f"Unknown weighting method: {method}")
    except Exception as e:
        logging.error(f"Error in weight calculation for method '{method}': {e}")
        return pd.Series(np.ones(n_assets) / n_assets, index=tickers)

def display_ma_deviation(history):
    st.subheader("Price Deviation from Moving Averages")

    if len(history) < 200:
        st.warning("Not enough data for Moving Average analysis (requires 200 days).")
        return

    try:
        price = history['Close'].iloc[-1]
        ma20 = history['Close'].rolling(window=20).mean().iloc[-1]
        ma50 = history['Close'].rolling(window=50).mean().iloc[-1]
        ma200 = history['Close'].rolling(window=200).mean().iloc[-1]
        std20 = history['Close'].rolling(window=20).std().iloc[-1]

        if pd.isna(ma20) or pd.isna(ma50) or pd.isna(ma200) or pd.isna(std20):
             st.warning("Could not calculate all moving average components.")
             return

        upper_band = ma20 + 2 * std20
        lower_band = ma20 - 2 * std20

        if price > upper_band:
            status = "Overbought"
            status_color = "red"
        elif price < lower_band:
            status = "Oversold"
            status_color = "green"
        else:
            status = "Neutral"
            status_color = "darkgray"

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=price,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"<b>{status}</b>", 'font': {'size': 20}},
            number={'prefix': "$", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [ma200 * 0.95, ma20 * 1.05], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': status_color, 'thickness': 0.3}, 'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "gray",
                'steps': [{'range': [ma200 * 0.95, ma50], 'color': 'rgba(0, 255, 0, 0.2)'}, {'range': [ma50, ma20 * 1.05], 'color': 'rgba(255, 0, 0, 0.2)'}],
                'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.9, 'value': price}
            }
        ))
        fig.add_trace(go.Scatter(
            x=[0.25, 0.5, 0.75], y=[-0.2, -0.2, -0.2],
            text=[f"MA20: ${ma20:.2f}", f"MA50: ${ma50:.2f}", f"MA200: ${ma200:.2f}"],
            mode="text", textfont=dict(size=10)
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying MA deviation chart: {e}")

def get_regression_metrics(history, period=126):
    if len(history) < period: return np.nan, "N/A"
    y = np.log(history['Close'].iloc[-period:])
    x = np.arange(len(y))
    slope, intercept, _, p_value, _ = linregress(x, y)
    predicted_y = slope * x + intercept
    residuals = y - predicted_y
    std_dev_from_reg = np.std(residuals)
    if p_value < 0.05:
        trend_str = "Positive" if slope > 0 else "Negative"
    else:
        trend_str = "Neutral"
    return std_dev_from_reg, trend_str

def get_daily_risk_range(history):
    if len(history) < 15: return np.nan, np.nan, np.nan, np.nan
    last_close = history['Close'].iloc[-1]
    prev_close = history['Close'].iloc[-2]
    high, low, close = history['High'], history['Low'], history['Close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean().iloc[-1]
    risk_low, risk_high = last_close - atr, last_close + atr
    pct_change = (last_close - prev_close) / prev_close * 100
    return risk_low, risk_high, last_close, pct_change

def display_momentum_bar(ticker_symbol, history):
    st.subheader("Dual-Scale Momentum (14-Day | 14-Hour)")
    rsi_14d = 50.0
    if len(history) >= 15:
        delta_d = history['Close'].diff()
        gain_d = (delta_d.where(delta_d > 0, 0)).rolling(window=14).mean()
        loss_d = (-delta_d.where(delta_d < 0, 0)).rolling(window=14).mean()
        if not pd.isna(loss_d.iloc[-1]) and loss_d.iloc[-1] != 0:
            rs_d = gain_d.iloc[-1] / loss_d.iloc[-1]
            rsi_14d = 100 - (100 / (1 + rs_d))
        elif not pd.isna(loss_d.iloc[-1]) and loss_d.iloc[-1] == 0 and gain_d.iloc[-1] > 0:
            rsi_14d = 100

    rsi_14h, has_hourly = 50.0, False
    try:
        hourly_hist = yf.Ticker(ticker_symbol).history(period="60d", interval="1h", auto_adjust=True)
        if not hourly_hist.empty and len(hourly_hist) >= 15:
            delta_h = hourly_hist['Close'].diff()
            gain_h = (delta_h.where(delta_h > 0, 0)).rolling(window=14).mean()
            loss_h = (-delta_h.where(delta_h < 0, 0)).rolling(window=14).mean()
            if not pd.isna(loss_h.iloc[-1]) and loss_h.iloc[-1] != 0:
                rs_h = gain_h.iloc[-1] / loss_h.iloc[-1]
                rsi_14h = 100 - (100 / (1 + rs_h))
            elif not pd.isna(loss_h.iloc[-1]) and loss_h.iloc[-1] == 0 and gain_h.iloc[-1] > 0:
                rsi_14h = 100
            has_hourly = True
    except Exception as e:
        logging.warning(f"Could not fetch hourly data for {ticker_symbol}: {e}")

    col1, col2 = st.columns(2)
    col1.metric("14-Day Trend (The Bar)", f"{rsi_14d:.1f}", help="RSI > 50 is bullish.")
    if has_hourly:
        col2.metric("14-Hour Pressure (The Marker)", f"{rsi_14h:.1f}", help="RSI > 50 indicates short-term upward pressure.")
    else:
        col2.info("Hourly data not available.")

    bar_color = "#04AA6D" if rsi_14d > 50 else "#FA3F46"
    fig = go.Figure()
    fig.add_trace(go.Bar(y=['RSI'], x=[rsi_14d], orientation='h', marker_color=bar_color, marker_line_width=0, width=0.5, hoverinfo='none'))
    fig.add_shape(type="line", x0=50, y0=-0.5, x1=50, y1=0.5, line=dict(color="rgba(255, 255, 255, 0.3)", width=1))
    fig.add_shape(type="line", x0=35, y0=-0.5, x1=35, y1=0.5, line=dict(color="rgba(255, 255, 255, 0.3)", width=1, dash="dash"))
    fig.add_shape(type="line", x0=65, y0=-0.5, x1=65, y1=0.5, line=dict(color="rgba(255, 255, 255, 0.3)", width=1, dash="dash"))
    if has_hourly:
        fig.add_shape(type="line", x0=rsi_14h, y0=-0.5, x1=rsi_14h, y1=0.5, line=dict(color="white", width=3))
    fig.update_layout(xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False, zeroline=False), yaxis=dict(showticklabels=False, showgrid=False, zeroline=False), showlegend=False, plot_bgcolor='rgba(68, 68, 68, 0.5)', paper_bgcolor='rgba(0,0,0,0)', height=40, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Bar shows 14-day trend; white marker shows 14-hour pressure; dashed lines at RSI 35 and 65.")
# --- START: Individual Stock Dashboard & Financials Functions ---

def calculate_portfolio_relative_z_score(weighted_df, etf_histories, best_etf, period="3y", window=252, min_window=200):
    """Calculates the relative Z-score of the entire portfolio against its best-correlated ETF."""
    portfolio_prices = None

    # Check if weights are valid
    if 'Weight' not in weighted_df.columns or weighted_df['Weight'].sum() == 0:
        logging.warning("Invalid weights for portfolio Z-score calculation.")
        return np.nan, best_etf

    # Create a weighted, rebased price series for the portfolio
    for idx, row in weighted_df.iterrows():
        ticker = row['Ticker']
        weight = row['Weight']
        try:
            history = yf.Ticker(ticker).history(period=period, auto_adjust=True, interval="1d").tz_localize(None)
            if history.empty or 'Close' not in history.columns or len(history) < 2:
                continue

            # Rebase the price series to start at 100 to normalize scales
            rebased_prices = 100 * (history['Close'] / history['Close'].iloc[0])

            if portfolio_prices is None:
                portfolio_prices = rebased_prices * weight
            else:
                # Align and add the weighted series
                portfolio_prices = portfolio_prices.add(rebased_prices * weight, fill_value=0)

        except Exception as e:
            logging.error(f"Error getting history for {ticker} in portfolio Z-score calc: {e}")
            continue

    if portfolio_prices is None or portfolio_prices.empty:
        logging.error("Failed to construct portfolio price series for Z-score calculation.")
        return np.nan, best_etf

    portfolio_prices.dropna(inplace=True)
    if portfolio_prices.empty:
        logging.error("Portfolio price series is empty after dropping NaNs.")
        return np.nan, best_etf

    etf_prices_df = etf_histories.get(best_etf)
    if etf_prices_df is None or etf_prices_df.empty or len(etf_prices_df) < 2:
        logging.warning(f"No history for best ETF '{best_etf}' in portfolio Z-score calculation.")
        return np.nan, best_etf

    # Rebase ETF prices as well
    etf_prices = 100 * (etf_prices_df['Close'] / etf_prices_df['Close'].iloc[0])

    common_index = portfolio_prices.index.intersection(etf_prices.index)[-window:]
    if len(common_index) < min_window:
        logging.warning(f"Insufficient overlap for portfolio vs {best_etf}: {len(common_index)} days")
        return np.nan, best_etf

    aligned_portfolio = portfolio_prices.loc[common_index]
    aligned_etf = etf_prices.loc[common_index]

    if (aligned_etf == 0).any():
        logging.warning(f"Zero prices found in ETF {best_etf}, cannot calculate relative strength.")
        return np.nan, best_etf

    relative = aligned_portfolio / aligned_etf

    if not relative.isna().any() and (relative > 0).all() and np.isfinite(relative).all():
        z_score = calculate_volatility_adjusted_z_score(relative, period=len(common_index), ticker="Portfolio")
        logging.info(f"Portfolio Z-Score vs {best_etf}: {z_score:.4f}")
        return z_score, best_etf

    logging.warning("Invalid portfolio relative data")
    return np.nan, best_etf
def calculate_absolute_z_score_and_trend(daily_history):
    if daily_history.empty or len(daily_history) < 252: return np.nan, "NEUTRAL"
    prices_log = np.log(daily_history['Close'].iloc[-252:])
    x = np.arange(len(prices_log))
    slope, _, r_value, p_value, _ = linregress(x, prices_log)
    trend_status = "NEUTRAL"
    if p_value < 0.05 and abs(r_value) > 0.4:
        if slope > 0.0005: trend_status = "UP-TREND"
        elif slope < -0.0005: trend_status = "DOWN-TREND"
    prices = daily_history['Close'].iloc[-126:]
    if len(prices) < 126: return np.nan, trend_status
    m_trend = prices.rolling(window=126, min_periods=63).median().iloc[-1]
    mad = np.median(np.abs(prices - m_trend))
    if pd.isna(m_trend) or pd.isna(mad) or mad == 0: return np.nan, trend_status
    absolute_z_score = (prices.iloc[-1] - m_trend) / (mad / 0.6745)
    return absolute_z_score, trend_status

def display_signal_sigma_checklist(stock_data, daily_history):
    st.subheader("Signal Sigma Checklist")
    def display_checklist_item(label, is_passed, help_text=""):
        col1, col2 = st.columns([0.1, 0.9])
        with col1: st.markdown("✅" if is_passed else "❌")
        with col2: st.markdown(f"**{label}**", help=help_text)

    absolute_z_score, trend_status = calculate_absolute_z_score_and_trend(daily_history)
    relative_z_score = stock_data.get('Relative_Z_Score')
    best_factor = stock_data.get('Best_Factor', 'its benchmark')
    return_6m, return_1y = stock_data.get('Return_126d'), stock_data.get('Return_252d')
    f_score = stock_data.get('Piotroski_F-Score')
    op_leverage = stock_data.get('Operating_Leverage')
    rd_ratio = stock_data.get('RD_to_Gross_Profit_2Y_Avg', 0) # Default to 0 if not present

    st.markdown(f"**Current Trend Status: `{trend_status}`**")
    st.markdown("---")
    st.markdown("#### Primary Rule (Internal Trend Timing)")
    if trend_status == "UP-TREND":
        is_above_lower_channel = pd.notna(absolute_z_score) and absolute_z_score >= -1.0
        display_checklist_item("Price is Above Lower Channel (Absolute Z > -1)", is_above_lower_channel, f"Current Absolute Z-Score: {absolute_z_score:.2f}")
    elif trend_status == "DOWN-TREND":
        is_above_m_trend = pd.notna(absolute_z_score) and absolute_z_score >= 0.0
        display_checklist_item("Price is Recovering Above M-Trend (Absolute Z > 0)", is_above_m_trend, f"Current Absolute Z-Score: {absolute_z_score:.2f}")
    else: st.info("The stock is in a NEUTRAL trend. The primary timing rules do not apply.")
    st.markdown("---")
    st.markdown("#### Universal Screening Rules (Strength & Quality)")
    is_outperforming = pd.notna(relative_z_score) and relative_z_score > 0
    display_checklist_item("Outperforming Benchmark (Relative Z > 0)", is_outperforming, f"Relative Z-Score vs {best_factor}: {relative_z_score:.2f}")
    positive_6m_return = pd.notna(return_6m) and return_6m > 0
    display_checklist_item("6-Month Return is Positive", positive_6m_return, f"Current 6M Return: {return_6m:.1f}%")
    positive_1y_return = pd.notna(return_1y) and return_1y > 0
    display_checklist_item("1-Year Return is Positive", positive_1y_return, f"Current 1Y Return: {return_1y:.1f}%")
    strong_f_score = pd.notna(f_score) and f_score >= 5
    display_checklist_item("Piotroski F-Score >= 5", strong_f_score, f"Current: {f_score:.0f}")
    good_op_leverage = pd.notna(op_leverage) and op_leverage >= 1
    display_checklist_item("Operating Leverage >= 1", good_op_leverage, f"Current: {op_leverage:.2f}")
    high_rd_investment = pd.notna(rd_ratio) and rd_ratio > 0.25
    display_checklist_item("R&D / Gross Profit > 25%", high_rd_investment, f"Current: {rd_ratio*100:.1f}%")

def valuation_wizard(ticker_symbol, revenue_growth_rate, gross_margin_rate, op_ex_as_percent_of_sales, share_count_growth_rate, ev_to_ebitda_multiple, tax_rate):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        cashflow = ticker.cashflow
        last_revenue = financials.loc['Total Revenue'].iloc[0] if 'Total Revenue' in financials.index else 0
        last_ebitda = info.get('ebitda') or (financials.loc['EBIT'].iloc[0] + cashflow.loc['Depreciation And Amortization'].iloc[0])
        last_interest_expense = financials.loc['Interest Expense'].iloc[0] if 'Interest Expense' in financials.index else 0
        current_shares = info.get('sharesOutstanding')
        current_eps = info.get('trailingEps')
        total_debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else info.get('totalDebt', 0)
        cash = balance_sheet.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in balance_sheet.index else info.get('totalCash', 0)
        net_debt = total_debt - cash
        if not all(v is not None and v > 0 for v in [last_revenue, current_shares, last_ebitda]):
            return np.nan, np.nan, "Error: Could not retrieve essential base data."
        if current_eps is None: return np.nan, np.nan, "Error: Could not retrieve Current EPS."
        debt_to_ebitda = net_debt / last_ebitda if last_ebitda > 0 else float('inf')
        if debt_to_ebitda > 10.0:
            return np.nan, np.nan, f"Valuation Unreliable: Debt to EBITDA ratio ({debt_to_ebitda:.1f}x) is excessively high."
        y5_revenue = last_revenue * ((1 + revenue_growth_rate) ** 5)
        y5_ebitda = last_ebitda * ((1 + revenue_growth_rate) ** 5)
        y5_ebit = y5_ebitda - (info.get('depreciation', 0) * (1 + revenue_growth_rate)**5)
        y5_enterprise_value = y5_ebitda * ev_to_ebitda_multiple
        y5_net_debt = net_debt * ((1 + revenue_growth_rate / 4) ** 5)
        y5_equity_value = y5_enterprise_value - y5_net_debt
        if y5_equity_value <= 0:
            return np.nan, np.nan, "Valuation Failed: Model predicts debt will exceed enterprise value."
        y5_shares = current_shares * ((1 + share_count_growth_rate) ** 5)
        price_target = y5_equity_value / y5_shares if y5_shares > 0 else 0
        if current_eps <= 0:
            return price_target, np.nan, "Current EPS is non-positive; CAGR cannot be calculated."
        else:
            y5_ebt = y5_ebit - (last_interest_expense * (1 + revenue_growth_rate / 4)**5)
            y5_net_income = y5_ebt * (1 - tax_rate)
            y5_eps = y5_net_income / y5_shares if y5_shares > 0 else 0
            if y5_eps > 0 and current_eps > 0:
                 eps_cagr = (y5_eps / current_eps)**(1/5) - 1
                 return price_target, eps_cagr, "Calculation successful."
            else:
                 return price_target, np.nan, "Model predicts negative EPS in Year 5."
    except Exception as e:
        return np.nan, np.nan, f"An error occurred: {e}"

def display_valuation_wizard(ticker_symbol):
    st.subheader("Valuation Wizard (5-Year Forecast)")
    try:
        ticker = yf.Ticker(ticker_symbol)
        history = ticker.history(period="3y")
        financials = ticker.financials
        info = ticker.info
        rev_g = (financials.loc['Total Revenue'].pct_change(periods=-1).mean()) * 100 if 'Total Revenue' in financials.index else 5.0
        gm = (financials.loc['Gross Profit'].iloc[0] / financials.loc['Total Revenue'].iloc[0]) * 100 if 'Gross Profit' in financials.index and 'Total Revenue' in financials.index and financials.loc['Total Revenue'].iloc[0] > 0 else 50.0
        op_inc = financials.loc['Operating Income'].iloc[0] if 'Operating Income' in financials.index else 0
        rev = financials.loc['Total Revenue'].iloc[0] if 'Total Revenue' in financials.index else 0
        opex_r = ((rev - op_inc) / rev) * 100 if rev > 0 else 30.0
        hist_ev_ebitda = info.get('enterpriseToEbitda', 15)
    except Exception as e:
        st.warning(f"Could not fetch historical data for context: {e}.")
        rev_g, gm, opex_r, hist_ev_ebitda = 5.0, 50.0, 30.0, 15.0

    with st.expander("Step 1: Set Your Fundamental Assumptions", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            rev_growth = st.slider("5Y Revenue Growth Rate (%)", -10.0, 40.0, round(float(rev_g or 5.0), 1), 0.5, key=f"rev_{ticker_symbol}")
            gross_margin = st.slider("5Y Gross Profit Margin (%)", 0.0, 100.0, round(float(gm or 50.0), 1), 0.5, key=f"gm_{ticker_symbol}")
            op_ex_ratio = st.slider("Operating Expenses as % of Sales", 0.0, 100.0, round(float(opex_r or 30.0), 1), 0.5, key=f"opex_{ticker_symbol}")
        with col2:
            ev_ebitda = st.slider("Terminal EV/EBITDA Multiple", 5.0, 40.0, round(float(hist_ev_ebitda or 15.0), 1), 0.5, key=f"ev_{ticker_symbol}")
            shares_growth = st.slider("Annual Share Count Growth (%)", -5.0, 5.0, -1.0, 0.1, key=f"shares_{ticker_symbol}")
            tax_rate_input = st.slider("Effective Tax Rate (%)", 15.0, 35.0, 21.0, 0.5, key=f"tax_{ticker_symbol}")

    if st.button("Calculate Fundamental Price Target", type="primary", key=f"calc_{ticker_symbol}"):
        with st.spinner("Running model..."):
            price_target, eps_cagr, commentary = valuation_wizard(ticker_symbol, rev_growth/100.0, gross_margin/100.0, op_ex_ratio/100.0, shares_growth/100.0, ev_ebitda, tax_rate_input/100.0)
        st.subheader("Model Results")
        if price_target and np.isfinite(price_target):
            current_price = history['Close'].iloc[-1]
            upside_pct = ((price_target - current_price) / current_price) * 100 if current_price > 0 else 0
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("Fundamental Price Target (5Y)", f"${price_target:,.2f}")
            res_col2.metric("Potential Upside", f"{upside_pct:.1f}%", delta=f"{upside_pct:.1f}%" if abs(upside_pct) > 0.1 else None)
            if eps_cagr and np.isfinite(eps_cagr):
                res_col3.metric("Implied 5Y EPS Growth (CAGR)", f"{eps_cagr:.2%}")
            else:
                res_col3.info(commentary)
        else:
            st.error(f"Valuation Failed. Reason: {commentary}")

# --- FIX: Replaced this entire function to be more robust against NaN data and fix argument passing ---
def get_correlated_stocks(selected_ticker, returns_dict, results_df, top_n=10):
    """
    Finds other tickers most correlated with the selected ticker. This version is
    robust against missing or non-overlapping return data.
    """
    if selected_ticker not in returns_dict or len(returns_dict) < 2:
        logging.warning(f"Correlation check failed: {selected_ticker} not in returns_dict or not enough tickers.")
        return pd.DataFrame()

    try:
        all_returns_df = pd.concat(returns_dict, axis=1)
    except Exception as e:
        logging.error(f"Failed to concat returns_dict: {e}")
        return pd.DataFrame()

    recent_returns = all_returns_df.tail(90)

    # Fill missing values with 0.0 to make correlation calculation robust
    recent_returns_filled = recent_returns.fillna(0.0)

    variances = recent_returns_filled.var()
    valid_columns = variances[variances > 1e-9].index
    recent_returns_final = recent_returns_filled[valid_columns]

    if selected_ticker not in recent_returns_final.columns:
        logging.warning(f"{selected_ticker} has no valid return data in the last 90 days.")
        return pd.DataFrame()

    corr_matrix = recent_returns_final.corr()

    if selected_ticker not in corr_matrix:
        return pd.DataFrame()

    correlations_to_selected = corr_matrix[selected_ticker].drop(selected_ticker, errors='ignore')

    if correlations_to_selected.empty:
        logging.warning(f"No other valid stocks to correlate with {selected_ticker}.")
        return pd.DataFrame()

    corr_df = pd.DataFrame(correlations_to_selected).rename(columns={selected_ticker: 'Correlation'})

    required_cols = ['Ticker', 'Best_Factor', 'Relative_Z_Score']
    if all(col in results_df.columns for col in required_cols):
        additional_info = results_df[required_cols].set_index('Ticker')
        corr_df = corr_df.join(additional_info)
        corr_df = corr_df.rename(columns={'Best_Factor': 'Benchmark'})
        corr_df['Benchmark'] = corr_df['Benchmark'].fillna("N/A")
    else:
        # Create placeholder columns if the main df is missing them
        corr_df['Relative_Z_Score'] = np.nan
        corr_df['Benchmark'] = "N/A"

    # Sort by the absolute value of 'Relative_Z_Score' and handle potential NaNs
    corr_df = corr_df.reindex(corr_df['Relative_Z_Score'].abs().fillna(0).sort_values(ascending=False).index)

    return corr_df.head(top_n)

# --- FIX: Corrected the call to `get_correlated_stocks` ---
def display_stock_dashboard(ticker_symbol, results_df, returns_dict, etf_histories):
    """Orchestrator function to display the entire individual stock dashboard."""
    st.header(f"🔬 Detailed Dashboard for {ticker_symbol}")
    try:
        daily_history = yf.Ticker(ticker_symbol).history(period="3y", auto_adjust=True, interval="1d")
        if daily_history.empty:
            st.warning("Could not fetch detailed daily history for this ticker.")
            return

        stock_data = results_df[results_df['Ticker'] == ticker_symbol].iloc[0].to_dict()
    except Exception as e:
        st.error(f"Error fetching data for dashboard: {e}")
        return

    if 'display_signal_sigma_checklist' in globals():
        display_signal_sigma_checklist(stock_data, daily_history)
        st.divider()

    col1, col2 = st.columns([1.2, 0.8])
    with col1:
        display_ma_deviation(daily_history)

        c1_tech, c2_tech, c3_tech = st.columns(3)
        with c1_tech:
            std_dev_reg, trend_str = get_regression_metrics(daily_history)
            st.metric("Std Dev From Trend", f"{std_dev_reg:.4f}")
        with c2_tech:
            st.metric("Medium-Term Trend", trend_str)
        with c3_tech:
            hurst_value = stock_data.get('Hurst_Exponent')
            if pd.notna(hurst_value):
                hurst_interpretation = "Trending" if hurst_value > 0.55 else "Mean-Reverting" if hurst_value < 0.45 else "Random"
                st.metric("Hurst Exponent", f"{hurst_value:.3f}", delta=hurst_interpretation, delta_color="off")
            else:
                st.metric("Hurst Exponent", "N/A")

        st.subheader("Daily Risk Range (ATR-based)")
        risk_low, risk_high, last_price, pct_change = get_daily_risk_range(daily_history)
        if not pd.isna(risk_low):
            c1_atr, c2_atr, c3_atr = st.columns(3)
            c1_atr.metric("Low", f"${risk_low:,.2f}")
            c2_atr.metric("Last", f"${last_price:,.2f}", f"{pct_change:.2f}%")
            c3_atr.metric("High", f"${risk_high:,.2f}")
        else:
            st.info("Not enough data for ATR calculation.")

        display_momentum_bar(ticker_symbol, daily_history)

    # --- THE FIX IS HERE ---
    with col2:
        st.subheader(f"Most Correlated Stocks (90d)")

        # 1. Call the function with the CORRECT arguments. No 'etf_histories'.
        correlated_stocks_df = get_correlated_stocks(ticker_symbol, returns_dict, results_df)

        if not correlated_stocks_df.empty:
            # 2. Display the CORRECT and CLEANED columns from the function above.
            st.dataframe(
                correlated_stocks_df[['Correlation', 'Relative_Z_Score', 'Benchmark']],
                use_container_width=True
            )
        else:
            st.write("No correlated stocks found.")

        if 'display_valuation_wizard' in globals():
            st.divider()
            display_valuation_wizard(ticker_symbol)

################################################################################
# SECTION 2: MAIN APPLICATION LOGIC
################################################################################
# --- FIX: THIS ENTIRE `main` FUNCTION HAS BEEN REWRITTEN FOR CORRECTNESS AND LOGIC ---
def main():
    st.title("Quantitative Portfolio Analysis")
    st.sidebar.header("Controls")
    if st.sidebar.button("Clear Cache & Re-run All", type="primary"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.subheader("Portfolio Construction")
    weighting_method_ui = st.sidebar.selectbox(
        "Portfolio Weighting Method",
        ["Equal Weight", "Inverse Volatility", "Log Log Sharpe Optimized", "Factor-Mimicking (Momentum)", "Alpha Orthogonal"]
    )
    new_factor = st.sidebar.selectbox(
        "Add & Trade a New Factor (FMP)",
        ["None", "Value (IVE)", "Growth (IVW)", "Quality (QUAL)", "Vision (Synthetic)"]
    )
    corr_window = st.sidebar.slider("Correlation Window (days)", min_value=30, max_value=180, value=90, step=30)

    # --- Data Fetching and Processing ---
    with st.spinner("Fetching ETF histories..."):
        etf_histories = fetch_all_etf_histories(etf_list)
    st.success("ETF histories loaded.")

    with st.spinner(f"Processing {len(tickers)} tickers... This may take several minutes."):
        results_df, failed_tickers, returns_dict = process_tickers(tickers, etf_histories, sector_etf_map)

    if results_df.empty:
        st.error("Fatal Error: No tickers could be processed.")
        st.stop()
    st.success(f"Successfully processed {len(results_df)} tickers.")
    if failed_tickers:
        st.expander("Show Failed Tickers").warning(f"{len(failed_tickers)} tickers failed: {', '.join(failed_tickers)}")

    # --- NEW: AUTOMATIC WEIGHTING BASED ON MULTI-HORIZON COEFFICIENT STABILITY ---
    st.sidebar.subheader("Automatic Factor Weighting")
    with st.spinner("Analyzing factor stability across multiple time horizons..."):
        # 1. Define the time horizons to test
        time_horizons = {
            "1M": "Return_21d",
            "3M": "Return_63d",
            "6M": "Return_126d",
            "12M": "Return_252d",
        }

        valid_metric_cols = [c for c in results_df.columns if pd.api.types.is_numeric_dtype(results_df[c]) and 'Return' not in c and c not in ['Ticker', 'Name', 'Score']]
        stability_results = {}

        # 2. Loop through each horizon, calculate pure returns, simulate history, and analyze stability
        for horizon_label, target_column in time_horizons.items():
            if target_column in results_df.columns:
                pure_returns_today = calculate_pure_returns(results_df, valid_metric_cols, target=target_column)
                if not pure_returns_today.empty:
                    historical_pure_returns = simulate_historical_pure_returns(pure_returns_today)
                    stability_df = analyze_coefficient_stability(historical_pure_returns)
                    stability_results[horizon_label] = stability_df

        # 3. Aggregate the stability results from all horizons to find the most consistent factors
        all_possible_metrics = list(default_weights.keys())
        auto_weights, rationale_df = aggregate_stability_and_set_weights(
            stability_results, all_possible_metrics, REVERSE_METRIC_NAME_MAP
        )

    with st.sidebar.expander("View Factor Stability Rationale", expanded=True):
        st.write("Weights are driven by a factor's **average performance** and **consistency** across 1, 3, 6, and 12-month return horizons. Higher scores are better.")
        st.dataframe(
            rationale_df[['avg_sharpe_coeff', 'consistency_score', 'horizons_present', 'Final_Weight']].loc[rationale_df['Final_Weight'] > 0.1].sort_values('Final_Weight', ascending=False),
            column_config={
                "avg_sharpe_coeff": st.column_config.NumberColumn("Avg Sharpe", help="Average signal-to-noise ratio of the factor's coefficient."),
                "consistency_score": st.column_config.ProgressColumn("Consistency", help="How consistently the factor's signal had the same sign across all horizons.", format="%.2f", min_value=0, max_value=1),
                "horizons_present": st.column_config.NumberColumn("Present In", help="Number of time horizons where this factor was significant."),
                "Final_Weight": st.column_config.NumberColumn("Weight %", format="%.2f")
            }
        )

    user_weights = auto_weights
    # --- END OF AUTOMATION BLOCK ---

    # --- Scoring Block ---
    raw_score = pd.Series(0.0, index=results_df.index)
    for long_name, weight in user_weights.items():
        if weight > 0:
            short_name = REVERSE_METRIC_NAME_MAP.get(long_name)
            if short_name in results_df.columns and short_name in rationale_df.index:
                rank_series = results_df[short_name].rank(pct=True)

                # *** CRITICAL FIX ***
                # Use the SIGN of the AGGREGATED sharpe coefficient to determine if lower is better
                if rationale_df.loc[short_name, 'avg_sharpe_coeff'] < 0:
                    rank_series = 1 - rank_series # Invert the rank for factors that are better when low (e.g., P/E ratio)

                raw_score += rank_series.fillna(0.5) * weight

    def z_score(series): return (series - series.mean()) / (series.std() if series.std() > 0 else 1)
    results_df['Score'] = z_score(raw_score)
    top_15_df = results_df.sort_values('Score', ascending=False).head(15).copy()
    top_15_tickers = top_15_df['Ticker'].tolist()

    # --- The rest of the main function remains unchanged ---

    st.header("📈 Portfolio Overview")
    if not top_15_tickers:
        st.warning("No stocks for portfolio construction.")
        st.stop()

    portfolio_returns_df = pd.DataFrame(returns_dict).reindex(columns=top_15_tickers).dropna(how='all')
    _, cov_matrix = calculate_correlation_matrix(top_15_tickers, returns_dict, window=corr_window)
    cov_matrix = cov_matrix.loc[top_15_tickers, top_15_tickers]

    momentum_factor_returns = etf_histories['MTUM']['Close'].pct_change().dropna()
    common_idx = portfolio_returns_df.index.intersection(momentum_factor_returns.index)
    aligned_returns = portfolio_returns_df.loc[common_idx].copy()
    aligned_momentum = momentum_factor_returns.loc[common_idx].copy()
    
    # --- FIX: Replace inf/-inf and fillna without inplace ---
    aligned_returns = aligned_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    aligned_momentum = aligned_momentum.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    betas = pd.DataFrame(index=top_15_tickers, columns=['Momentum_Beta'])
    for ticker in top_15_tickers:
        try:
            model = LinearRegression().fit(aligned_momentum.values.reshape(-1, 1), aligned_returns[ticker].values)
            betas.loc[ticker, 'Momentum_Beta'] = model.coef_[0]
        except Exception: betas.loc[ticker, 'Momentum_Beta'] = 1.0

    if new_factor != "None":
        st.subheader(f"FMP for: {new_factor}")
        factor_map = {"Value (IVE)": "IVE", "Growth (IVW)": "IVW", "Quality (QUAL)": "QUAL", "Vision (Synthetic)": "VISION_SYNTHETIC"}
        key = factor_map.get(new_factor)
        if key in etf_histories:
            factor_ts = etf_histories[key]['Close'].pct_change().dropna()
            p_weights = calculate_fmp_weights(aligned_returns, factor_ts, cov_matrix, existing_factors_returns=aligned_momentum.to_frame())

            weights_df = p_weights.reset_index()
            weights_df.columns = ['Ticker', 'FMP Weight']
            weights_df = pd.merge(weights_df, top_15_df[['Ticker', 'Name']], on='Ticker', how='left')
            display_cols = ['Ticker', 'Name', 'FMP Weight']
            st.dataframe(
                weights_df.sort_values("FMP Weight", key=abs, ascending=False)[display_cols],
                use_container_width=True
            )
    else:
        st.subheader(f"Portfolio Weights ({weighting_method_ui})")
        method_map = {"Equal Weight": "equal", "Inverse Volatility": "inv_vol", "Log Log Sharpe Optimized": "log_log_sharpe"}
        if weighting_method_ui == "Factor-Mimicking (Momentum)": p_weights = calculate_weights(aligned_returns, method="fmp", cov_matrix=cov_matrix, factor_returns=aligned_momentum)
        elif weighting_method_ui == "Alpha Orthogonal": p_weights = calculate_weights(aligned_returns, method="alpha_orthogonal", betas=betas)
        else: p_weights = calculate_weights(aligned_returns, method=method_map.get(weighting_method_ui, "equal"), cov_matrix=cov_matrix)

        weights_df = p_weights.reset_index()
        weights_df.columns = ['Ticker', 'Weight']
        weights_df = pd.merge(weights_df, top_15_df[['Ticker', 'Name']], on='Ticker', how='left')
        display_cols = ['Ticker', 'Name', 'Weight']
        st.dataframe(
            weights_df.sort_values("Weight", ascending=False)[display_cols],
            use_container_width=True
        )

    weighted_df_calc = pd.DataFrame()
    if 'weights_df' in locals():
        if 'Weight' in weights_df.columns: weighted_df_calc = weights_df[['Ticker', 'Weight']].copy()
        elif 'FMP Weight' in weights_df.columns: weighted_df_calc = weights_df[['Ticker', 'FMP Weight']].rename(columns={'FMP Weight': 'Weight'}).copy()

    if not weighted_df_calc.empty:
        corrs = calculate_portfolio_factor_correlations(weighted_df_calc, etf_histories)
        best_etf, best_corr = (corrs.index[0], corrs.iloc[0]) if not corrs.empty else ('SPY', np.nan)
        z, _ = calculate_portfolio_relative_z_score(weighted_df_calc, etf_histories, best_etf)
        st.write(f"**Top-Correlated ETF:** `{best_etf}` (Correlation: {best_corr:.4f})")
        st.write(f"**Portfolio Relative Z-Score vs {best_etf}:** {z:.4f}")

    if 'p_weights' in locals() and p_weights is not None and not p_weights.empty:
        aligned_w = p_weights.reindex(aligned_returns.columns).fillna(0)
        final_returns = (aligned_returns * aligned_w).sum(axis=1)
        scores = top_15_df.set_index('Ticker').reindex(aligned_returns.columns)['Score'].fillna(0)
        alpha_weights = scores / scores.sum() if scores.sum() != 0 else pd.Series(1/len(scores), index=scores.index)
        forecast_ts_unlagged = (aligned_returns * alpha_weights).sum(axis=1)
        lagged_forecast_ts = forecast_ts_unlagged.shift(1)
        malv, _ = calculate_mahalanobis_metrics(aligned_returns, cov_matrix)
        ic, ir = calculate_information_metrics(lagged_forecast_ts, final_returns)
        col1, col2, col3 = st.columns(3)
        col1.metric("Precision Matrix Quality (MALV)", f"{malv:.4f}", help=f"Expected: {2/len(top_15_tickers):.4f}")
        col2.metric("Information Coefficient (IC)", f"{ic:.4f}", help="Lagged correlation of alpha vs returns.")
        col3.metric("Information Ratio (IR)", f"{ir:.4f}", help="Risk-adjusted return (Sharpe).")

    # --- Detailed Report Tabs ---
    st.header("📊 Detailed Reports")
    st.sidebar.divider(); st.sidebar.header("Individual Stock Analysis")

    options = sorted(results_df['Ticker'].unique().tolist())
    default_ticker = top_15_df['Ticker'].iloc[0] if not top_15_df.empty and top_15_df['Ticker'].iloc[0] in options else options[0] if options else None

    selected_ticker = st.sidebar.selectbox(
        "Select a Ticker",
        options=options,
        index=options.index(default_ticker) if default_ticker else 0
    )

    tab1, tab2, tab3 = st.tabs(["🔬 Stock Dashboard & Financials", "🎛️ Factor Analysis", "📄 Full Data Table"])
    with tab1:
        if selected_ticker:
            display_stock_dashboard(selected_ticker, results_df, returns_dict, etf_histories)
            display_deep_dive_data(selected_ticker)
    with tab2:
        st.subheader("Pure Factor Returns (Aggregated & Individual Horizons)")
        # Display the main aggregated rationale first
        st.write("#### Aggregated Factor Performance")
        st.dataframe(rationale_df)

        # Then allow drilling down into individual horizons
        for horizon_label, stability_df in stability_results.items():
             with st.expander(f"Details for {horizon_label} Horizon (Target: {time_horizons[horizon_label]})"):
                 if not stability_df.empty:
                     st.dataframe(stability_df)
                 else:
                     st.warning(f"No significant factors found for the {horizon_label} horizon.")
    with tab3:
        st.subheader("Full Processed Data Table")
        st.dataframe(results_df)

if __name__ == "__main__":
    main()
