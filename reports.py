import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set plot style
sns.set_style("whitegrid")

# Load the dataset
file_path = 'Airbnb_data_cleaned.csv'
try:
    df = pd.read_csv(file_path)

    # --- Initial Data Inspection ---
    print("Data Info:")
    df.info()
    print("\nData Head:")
    print(df.head())
    print("\nNumeric Columns Description:")
    print(df.describe())

    # --- Feature Engineering ---
    # Assuming 'sales' is the number of bookings/days, revenue is price * sales.
    if 'price' in df.columns and 'sales' in df.columns:
        df['revenue'] = df['price'] * df['sales']
    else:
        print("Error: 'price' or 'sales' column not found. Cannot calculate revenue.")
        exit()

    # --- Visualization ---
    
    # 1. Price Distribution (Capped for better visibility)
    price_cap = df['price'].quantile(0.95)
    plt.figure(figsize=(10, 6))
    sns.histplot(df[df['price'] < price_cap]['price'], bins=50, kde=True)
    plt.title(f'Distribution of Listing Price (Capped at 95th Percentile: ${price_cap:.2f}$)')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.savefig('price_distribution_capped.png')
    plt.close()

    # 2. Revenue Distribution (Capped for better visibility)
    revenue_cap = df['revenue'].quantile(0.95)
    plt.figure(figsize=(10, 6))
    sns.histplot(df[df['revenue'] < revenue_cap]['revenue'], bins=50, kde=True)
    plt.title(f'Distribution of Calculated Revenue (Capped at 95th Percentile)')
    plt.xlabel('Revenue')
    plt.ylabel('Frequency')
    plt.savefig('revenue_distribution_capped.png')
    plt.close()

    # 3. Correlation Heatmap
    # Select key numeric features for correlation
    numeric_cols = ['price', 'sales', 'revenue', 'host_total_listings', 'total_reviewers',
                    'accommodates', 'bathrooms', 'bedrooms', 'beds',
                    'host_response_rate', 'host_acceptance_rate']
    
    # Filter columns that actually exist in the dataframe
    existing_numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    plt.figure(figsize=(12, 10))
    # Calculate correlation only on existing numeric columns
    corr = df[existing_numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap of Key Numeric Features')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()

    # 4. Average Revenue by Room Type
    if 'room_type' in df.columns:
        plt.figure(figsize=(10, 6))
        # Group by room_type, calculate mean revenue, and sort
        avg_revenue_room_type = df.groupby('room_type')['revenue'].mean().sort_values(ascending=False)
        sns.barplot(x=avg_revenue_room_type.index, y=avg_revenue_room_type.values, order=avg_revenue_room_type.index)
        plt.title('Average Revenue by Room Type')
        plt.xlabel('Room Type (Encoded)')
        plt.ylabel('Average Revenue')
        plt.savefig('avg_revenue_by_room_type.png')
        plt.close()

    # 5. Average Revenue by Top 10 Cities
    if 'city' in df.columns:
        top_cities = df['city'].value_counts().nlargest(10).index
        df_top_cities = df[df['city'].isin(top_cities)]
        
        plt.figure(figsize=(15, 8))
        # Group by city, calculate mean revenue, and sort
        avg_revenue_city = df_top_cities.groupby('city')['revenue'].mean().sort_values(ascending=False)
        sns.barplot(x=avg_revenue_city.index, y=avg_revenue_city.values, order=avg_revenue_city.index)
        plt.title('Average Revenue for Top 10 Cities')
        plt.xlabel('City')
        plt.ylabel('Average Revenue')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('avg_revenue_by_top_cities.png')
        plt.close()

    # 6. Price vs. Sales (Demand Curve Proxy)
    # Use a sample to avoid overplotting
    df_sample = df.sample(n=min(5000, len(df)), random_state=42)
    
    plt.figure(figsize=(12, 7))
    sns.scatterplot(x='price', y='sales', data=df_sample[df_sample['price'] < price_cap], alpha=0.3)
    # Add a trend line
    sns.regplot(x='price', y='sales', data=df_sample[df_sample['price'] < price_cap], 
                scatter=False, color='red', line_kws={'linestyle':'--'}, lowess=True)
    plt.title('Price vs. Sales (Demand) (Capped Price)')
    plt.xlabel('Price')
    plt.ylabel('Sales (Bookings/Days)')
    plt.savefig('price_vs_sales_scatter.png')
    plt.close()

    # 7. Price vs. Revenue (Revenue Optimization Curve)
    plt.figure(figsize=(12, 7))
    sns.scatterplot(x='price', y='revenue', data=df_sample[df_sample['price'] < price_cap], alpha=0.3)
    # Add a trend line (LOWESS) to see the curve
    sns.regplot(x='price', y='revenue', data=df_sample[df_sample['price'] < price_cap], 
                scatter=False, color='red', lowess=True, line_kws={'linestyle':'--'})
    plt.title('Price vs. Revenue (Capped Price)')
    plt.xlabel('Price')
    plt.ylabel('Revenue')
    plt.savefig('price_vs_revenue_scatter.png')
    plt.close()

    # 8. Average Revenue by Accommodates
    if 'accommodates' in df.columns:
        plt.figure(figsize=(12, 7))
        avg_revenue_accommodates = df.groupby('accommodates')['revenue'].mean().sort_values(ascending=False)
        sns.barplot(x=avg_revenue_accommodates.index, y=avg_revenue_accommodates.values, order=avg_revenue_accommodates.index)
        plt.title('Average Revenue by Number of Accommodates')
        plt.xlabel('Accommodates')
        plt.ylabel('Average Revenue')
        plt.savefig('avg_revenue_by_accommodates.png')
        plt.close()

    # 9. Average Revenue by Guest Favourite Status
    if 'guest_favourite' in df.columns:
        plt.figure(figsize=(8, 6))
        avg_revenue_guest_favourite = df.groupby('guest_favourite')['revenue'].mean().sort_values(ascending=False)
        sns.barplot(x=avg_revenue_guest_favourite.index, y=avg_revenue_guest_favourite.values, order=avg_revenue_guest_favourite.index)
        plt.title('Average Revenue by Guest Favourite Status')
        plt.xlabel('Is Guest Favourite')
        plt.ylabel('Average Revenue')
        plt.savefig('avg_revenue_by_guest_favourite.png')
        plt.close()

    print("\nAnalysis complete. Generated the following plots:")
    print("- price_distribution_capped.png")
    print("- revenue_distribution_capped.png")
    print("- correlation_heatmap.png")
    if 'room_type' in df.columns: print("- avg_revenue_by_room_type.png")
    if 'city' in df.columns: print("- avg_revenue_by_top_cities.png")
    print("- price_vs_sales_scatter.png")
    print("- price_vs_revenue_scatter.png")
    if 'accommodates' in df.columns: print("- avg_revenue_by_accommodates.png")
    if 'guest_favourite' in df.columns: print("- avg_revenue_by_guest_favourite.png")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

