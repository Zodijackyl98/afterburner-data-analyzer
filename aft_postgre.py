from main import PerformanceAnalyzer
from sqlalchemy import create_engine

path = r"C:\Python_Works\py\afterburner-data-analyzer\examples\texts\HMW_bf2042_RT.txt"

# Initialize analyzer
analyzer = PerformanceAnalyzer(path)

user = 'mert'
password = 'password'
db_name = 'afterburner'
address = 'localhost'
port = '5432'


engine = create_engine(f"postgresql://{user:s}:{password:s}@{address:s}:{port:s}/{db_name:s}")

# Load and process data
if analyzer.load_data():
    df = analyzer.df_final

    # Remove duplicate timestamps before saving to database
    df.drop_duplicates(subset=['format_time_aft'], keep='first')
    df.to_sql("bf2042", con=engine, if_exists='replace', index=True)