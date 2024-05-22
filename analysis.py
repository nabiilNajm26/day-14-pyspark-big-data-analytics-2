from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, sum as _sum, avg as _avg, round as _round, to_date, date_format, lit, concat, lpad
import os
import plotly.express as px
import plotly.graph_objects as go

# Initialize Spark session
spark = SparkSession.builder.appName('dibimbing').master('local').getOrCreate()

# Load datasets
calendar_df = spark.read.csv("data/calendar.csv", header=True, inferSchema=True)
customer_flight_activity_df = spark.read.csv("data/customer_flight_activity.csv", header=True, inferSchema=True)
customer_loyalty_history_df = spark.read.csv("data/customer_loyalty_history.csv", header=True, inferSchema=True)

# Inspect data
calendar_df.show(5)
customer_flight_activity_df.show(5)
customer_loyalty_history_df.show(5)

# Data Cleaning & Transformation
customer_flight_activity_df = customer_flight_activity_df.na.fill(0)
customer_loyalty_history_df = customer_loyalty_history_df.na.fill({'Salary': 0, 'customer_lifetime_value': 0})

# Join datasets if necessary for analysis
data_df = customer_flight_activity_df.join(customer_loyalty_history_df,
                                           customer_flight_activity_df["loyalty_number"] == customer_loyalty_history_df["loyalty_number"],
                                           how="inner")

# Inspect the joined dataset
data_df.printSchema()
data_df.show(5)

# Analysis 1: Total Flights per Year
total_flights_per_year = data_df.groupBy("year").agg(_sum("total_flights").alias("total_flights")).orderBy("year")
total_flights_per_year.show()

# Convert to Pandas for plotting
total_flights_per_year_df = total_flights_per_year.toPandas()

# Plot using Plotly
fig1 = px.line(total_flights_per_year_df, x='year', y='total_flights', title='Total Flights per Year')
fig1.update_layout(xaxis_title='Year', yaxis_title='Total Flights')
fig1.write_image("graph/total_flights_per_year.png")
fig1.show()

# Analysis 2: Average Distance Traveled per Loyalty Card Type
avg_distance_per_card = data_df.groupBy("loyalty_card").agg(_avg("distance").alias("avg_distance")).orderBy("loyalty_card")
avg_distance_per_card.show()

# Convert to Pandas for plotting
avg_distance_per_card_df = avg_distance_per_card.toPandas()

# Plot using Plotly
min_distance = avg_distance_per_card_df['avg_distance'].min()

fig2 = px.bar(avg_distance_per_card_df, x='loyalty_card', y='avg_distance', title='Average Distance Traveled per Loyalty Card Type')
fig2.update_layout(
    xaxis_title='Loyalty Card Type',
    yaxis_title='Average Distance',
    yaxis=dict(range=[min_distance - 10, avg_distance_per_card_df['avg_distance'].max() + 10])  # Set y-axis range
)
fig2.write_image("graph/avg_distance_per_card.png")
fig2.show()

# Analysis 3: Flight Activity Over Time by Loyalty Card Type
monthly_flight_activity = data_df.groupBy("year", "month", "loyalty_card").agg(_sum("total_flights").alias("total_flights")).orderBy("year", "month", "loyalty_card")
monthly_flight_activity.show()

# Combine year and month into a datetime column for easier plotting
monthly_flight_activity = monthly_flight_activity.withColumn(
    'date',
    to_date(concat(col('year'), lpad(col('month'), 2, '0'), lit('01')), 'yyyyMMdd')
)

# Convert to Pandas for plotting
monthly_flight_activity_df = monthly_flight_activity.toPandas()

# Pivot Data
pivot_df = monthly_flight_activity_df.pivot(index='date', columns='loyalty_card', values='total_flights').fillna(0)
pivot_df.reset_index(inplace=True)

# Plot using Plotly
fig3 = go.Figure()

for card in ['Aurora', 'Nova', 'Star']:
    fig3.add_trace(go.Scatter(x=pivot_df['date'], y=pivot_df[card], mode='lines+markers', name=card))

fig3.update_layout(title='Monthly Flight Activity Over Time by Loyalty Card Type',
                   xaxis_title='Date',
                   yaxis_title='Total Flights',
                   legend_title='Loyalty Card Type')
fig3.write_image("graph/monthly_flight_activity_by_loyalty_card.png")
fig3.show()

# Exporting the analysis results to CSV
output_dir = "analysis_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

total_flights_per_year.write.csv(os.path.join(output_dir, "total_flights_per_year.csv"), header=True)
avg_distance_per_card.write.csv(os.path.join(output_dir, "avg_distance_per_card.csv"), header=True)
monthly_flight_activity.write.csv(os.path.join(output_dir, "monthly_flight_activity.csv"), header=True)

print("Analysis and export complete.")