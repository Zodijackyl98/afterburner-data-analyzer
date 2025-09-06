from main import PerformanceAnalyzer
from sqlalchemy import create_engine

path = r"C:\Python_Works\py\afterburner-data-analyzer\examples\texts\HMW_MHW.txt"


# Initialize analyzer
analyzer = PerformanceAnalyzer(path)

user = 'mert'
password = 'password'
db_name = 'afterburner'
address = 'localhost'
port = '5432'


engine = create_engine(f"postgresql+psycopg2://{user:s}:{password:s}@{address:s}:{port:s}/{db_name:s}")

# Load and process data
if analyzer.load_data():
    df = analyzer.df_final
    # Remove duplicate timestamps before saving to database
    df.drop_duplicates(subset=['format_time_aft'], keep='first', inplace = True)
    df.to_sql("hmw_mhw", con=engine, if_exists='replace', index=True)