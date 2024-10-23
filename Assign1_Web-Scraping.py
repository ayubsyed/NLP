"""
This file scrapes review of the company Pure Filter from Trust Pilot
"""

# Importing libraries
import requests
import time
import pandas as pd
from bs4 import BeautifulSoup

# Assigning the URL to PureFilters reviews on TrustPilot
url = "https://ca.trustpilot.com/review/purefilters.ca"

# Some websites block non-browser requests, so including the User-Agent
# can help bypass that. The headers simulate a request coming from a browser.
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
        (KHTML, like Gecko) '
                  'Chrome/91.0.4472.124 Safari/537.36'
}

# Sending an HTTP GET request to download the HTML content from the URL
response = requests.get(url, headers=headers)

# Checking if the request was successful.
# A successful request has a status code of 200
if response.status_code == 200:
    print("Successfully downloaded the HTML content!")
else:
    print(f"Failed to download HTML. Status code: {response.status_code}")

# Parsing the HTML using BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Finding the element that contains the total number of reviews.
# This will be the <p> tag with a specific class name.
total_reviews_element = soup.find(
    name='p',
    attrs={'class': 'typography_body-l__KUYFJ typography_appearance-default__AAY17'}
)

# Extracting the number of reviews as text and cleaning it.
# Assuming the content is a number with commas (e.g., "1,234").
if total_reviews_element:
    total_reviews_text = total_reviews_element.contents[0].replace(',', '')
    total_reviews = int(total_reviews_text)
    print(f"Total number of reviews: {total_reviews}")
else:
    print("Could not find the total number of reviews on the page.")

# Initialize an empty list to store review data for each page
rows_list = []

# Defining the base URL for Trustpilot
base_url = "https://ca.trustpilot.com"
start_url = "/review/purefilters.ca"

# Initializing a loop to iterate over multiple pages
current_url = base_url + start_url
while current_url:
    # Sending a GET request to the current page
    response = requests.get(current_url, headers=headers)

    # Checking if the request was successful
    if response.status_code == 200:
        # Parsing the HTML content with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extracting the reviews on the current page
        print(f"Scraping reviews from: {current_url}")

        # Locating all the review cards on the page
        reviews = soup.find_all('section', 
                                class_='styles_reviewContentwrapper__zH_9M')
        print(len(reviews))
        print(reviews)

        # Looping through each review found and extracting the relevant details
        for review in reviews:
            # Extracting the review body (the text of the review)
            print(review)
            review_title = review.find('h2')
            review_body = review.find('p', attrs={
                'data-service-review-text-typography': 'true'
                })
            print(review_body)
            review_content = (
                f'{review_title.get_text(strip=True) if review_title else ""} '
                f'{review_body.get_text(strip=True) if review_body else ""}'
            )

            # Extracting the rating value (1 to 5 stars)
            # The rating is often stored in a div with a class that indicates
            # the star rating, or in a data attribute.
            rating_value = review.find('img')
            rating_value = rating_value.attrs['alt'].split()[1]

            # Extracting the date the review was published
            # Dates are often stored in a <time> tag, and we extract the 'datetime'
            # attribute to get the full date.
            if review.find('time'):
                date_published = review.find('time')['datetime']
            else:
                date_published = ""

            # Setting the company name (static for all reviews)
            company_name = "PureFilters"
            data = {
                "companyName": company_name,
                "datePublished": date_published,
                "ratingValue": rating_value,
                "reviewBody": review_content
            }
            rows_list.append(data)

        # Finding the "Next page" link
        next_page = soup.find_all(name="a", string="Next page")

        # Checking if the "Next page" link exists
        if next_page:
            # Debugging: Print the found tag to check its structure
            print("Next page link found:", next_page[0])

            # Checking if 'href' exists in the first result
            if 'href' in next_page[0].attrs:
                next_page_url = next_page[0]['href']
                current_url = base_url + next_page_url
                time.sleep(2)
            else:
                print("No href found in the 'Next page' link.")
                break
        else:
            # If no "Next page" link is found, stop the loop
            print("No more pages to scrape.")
            current_url = None
    else:
        print(f"Failed to retrieve page. Status code: {response.status_code}")
        break

# Creating a DataFrame with the required columns
df = pd.DataFrame(rows_list)

# Saving the DataFrame to a CSV file
df.to_csv('purefilters_reviews.csv', index=False)

print("Reviews have been successfully saved to purefilters_reviews.csv!")
