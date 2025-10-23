import pandas as pd

df = pd.read_csv("Airbnb_data.csv", low_memory=False)

def clean_data(dataset):

    dataset.drop(columns=['consumer','host since'], inplace=True)

    dataset=dataset.dropna()

    # Removing commas and converting to intergers to float

    dataset['price'] = dataset['price'].replace(',', '.', regex=True).astype(float)

    dataset['bathrooms'] = dataset['bathrooms'].replace(',', '.', regex=True).astype(float)

    dataset['host response rate'] = dataset['host response rate'].replace(',', '.', regex=True).astype(float)

    dataset['host acceptance rate'] = dataset['host acceptance rate'].replace(',', '.', regex=True).astype(float)

    dataset[['host Certification','guest favourite']] = dataset[['host Certification','guest favourite']].astype(bool)

    # Removing blank spaces from column names for uniformity

    dataset.rename(columns={
       'reply time': 'reply_time',
       'guest favourite': 'guest_favourite',
       'host Certification': 'host_certification',
       'host total listings count': 'host_total_listings',
       'total reviewers number': 'total_reviewers',
       'listing number': 'listing_number',
       'host response rate': 'host_response_rate',
       'host acceptance rate': 'host_acceptance_rate'
       }, inplace=True)

    return dataset


df_cleaned = clean_data(df)

df_cleaned.to_csv("Airbnb_data_cleaned.csv")
