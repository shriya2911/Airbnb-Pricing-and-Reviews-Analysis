# Airbnb Pricing & Guest Sentiment Analysis in Texas

A comprehensive Jupyter Notebook–driven analysis in Python leveraged Airbnb datasets from Austin, Dallas, and Fort Worth to uncover the primary factors influencing nightly rates and guest satisfaction. The workflow began with rigorous data cleansing—parsing and standardizing pricing and amenity information, imputing missing values, and removing statistical outliers—followed by targeted feature engineering to capture host tenure, availability dynamics, and property characteristics. Exploratory visualizations, spatial heatmaps, and NLP-based sentiment scoring distilled guest feedback into actionable insights, which informed the development of regression and ensemble models (Linear, Ridge, Lasso, Random Forest, XGBoost) that achieved an R² up to 0.71 on price prediction. The resulting recommendations guide pricing strategies across high-value neighborhoods, amenity investments to boost occupancy and RevPAR, and nuanced guest-centric messaging to enhance overall Net Promoter Scores.

## Objectives

- Quantify the impact of location, room/property type, and amenities on listing price
- Classify guest sentiment via NLP and sentiment analysis (VADER)
- Model price predictions using linear models, Random Forest, XGBoost, Ridge/Lasso
- Recommend data-driven strategies to boost occupancy, RevPAR, and guest NPS

## Data 

**Datasets:**
- Listings (Austin: 15,159×75; Dallas: 4,912×75; Fort Worth: 1,836×75) — pricing, availability, host metadata, geo-coordinates
- Reviews (Austin: 617,430; Dallas: 201,901; Fort Worth: 73,931) — reviewer ID/date, raw comments

**Ingestion:**

- Loaded raw CSVs into pandas DataFrames (`df_austin`, `df_dallas`, `df_fortworth`)
- Joined review scores and price from listings to reviews for enriched analysis

**Source** 

All raw Airbnb data for Austin, Dallas, and Fort Worth can be downloaded from the Inside Airbnb portal:  [Inside Airbnb – Get the Data](https://insideairbnb.com/get-the-data/)


## Data Cleaning 

- **Price**: Stripped “$”/commas; converted to numeric. Imputed nulls with mean price by `(bedrooms × city)`. Removed outliers via whisker limits (1.5× IQR)
- **Host Acceptance Rate**: Removed “%”; numeric conversion; filled missing with column mean
- **Bathrooms**: Parsed `bathrooms_text` via regex to extract counts; filled nulls accordingly
- **Beds & Bedrooms**: Grouped by `(bedrooms, city)` and `(room_type, city)` respectively; imputed medians to avoid skew and decimals
- **Neighbourhood_Cleansed (Fort Worth only)**: Nearest-neighbor imputation (scikit-learn NearestNeighbors) based on lat/long
- **Review Scores**: Replaced nulls with per-column means for consistency across metrics

- **Text (Comments):**
  - Lowercased, removed special characters, tokenized
  - Lemmatized and filtered custom + NLTK stopwords
  - Rebuilt into `cleaned_comments` column

## Feature Engineering

- **Property Type Mapping**: Collapsed 30+ values into 3 buckets—Alternative Accommodation, Outdoor & Adventure, Hotels/Resorts—to reduce cardinality
- **Amenities One-Hot**: Created binary flags for top amenities (e.g., Netflix, Hot Tub, AC, Security Cameras, Balcony)
- **Dropped Irrelevant**: Removed metadata columns (e.g., `listing_url`, `picture_url`, `host_name`, `last_scraped`) to cut noise.
- **Host Tenure**: Computed `host_duration_days` = `today’s date` − `host_since` (datetime conversion) 
- **Availability Ratios**: Derived `availability_30_ratio`, `availability_60_ratio`, `availability_90_ratio` as % available over each window, then dropped raw counts
- **Reviews Join**: Merged `price` and `review_scores_rating` from listings into reviews table for joint modeling

## Exploratory Data Analysis

- **Price Distribution:**
  - *Initial histogram:* heavy right skew, necessitating outlier treatment annotated-
  - *Post-cleaning histogram/boxplot (≤ $600):* interquartile $100–$200, outliers trimmed 
- **Room & Property Types:** Entire homes/apts command highest prices; shared rooms lowest; hotels/private midpoint 
- **Spatial Heatmaps:** Median-price choropleths by neighborhood for each city, identifying premium (e.g., 78704 in Austin) vs. budget zones
- **Review Text Analytics:**
  - *Unigrams/Bigrams/Trigrams:* Extracted n-grams to surface frequent themes (e.g., “great location,” “clean room”) 
  - *Word Cloud:* Visualized top tokens; confirmed overwhelmingly positive language
  - *Sentiment (VADER):* Classified comments into Positive (> 0.05), Neutral (±0.05), Negative (< −0.05); ~96% positive

- [EDA & Modeling Notebook](https://github.com/shriya2911/Capstone_project/blob/main/EDA%2BModeling.ipynb) – EDA, Modeling & Evaluation (Pricing Data)
- [Sentiment Analysis Notebook](https://github.com/shriya2911/Capstone_project/blob/main/Sentiment%20%2B%20Theme%20Classification.ipynb) - Sentiment & Text Analytics (Reviews Data) 

## Recommendations

- **Prioritize High-Value Neighborhoods:** Focus pricing strategies on premium ZIPs (e.g., 78704 in Austin; Districts 11 & 8 in Dallas; Districts 3 & 4 in Fort Worth) to capture 20–35% higher ADRs.
- **Leverage Private-Stay Demand:**“Entire home/apt” listings consistently command 25–40% price premiums—even in budget areas—so optimize inventory and marketing toward group/family segments.
- **Enhance Core Amenities & Specialty Features:** Ensure baseline offerings (Wi-Fi, AC, kitchen essentials) for occupancy, and add luxury add-ons (hot tubs, workspaces) to boost NPS and justify higher RevPAR.
- **Tailor Offerings by Topic Insights:** Listings tagged “Family/Group/Event Friendly” or “Amazing Amenities” out-perform others; highlight these themes in descriptions and adjust pricing accordingly.
**Dynamic Pricing & Emerging Markets:** Implement real-time, demand-driven pricing engines and explore underserved neighborhoods with low listing counts (e.g., Austin’s 78730) to capture new revenue streams.

## Data Limitations

- **Geographic Scope:** Analysis is confined to three Texas cities—Austin, Dallas, Fort Worth—so findings may not generalize to other markets.
- **Single-Platform Data:** All listings and reviews sourced exclusively from Airbnb; competitor platforms (VRBO, Booking.com) aren’t represented.
- **Rule-Based Text Classification:** Topic tagging relies on keyword rules (4% of comments remained untagged), which may miss nuances or emerging themes in guest feedback. annotated-
- **Imputation Choices:** Filling missing values (e.g., price by bedroom/city median; KNN for neighborhoods) could introduce bias, particularly in low-density areas with sparse data.

## Scope & Future Work

- **Advanced Topic Modeling:** Move beyond rule-based classification to LDA or BERTopic for richer, unsupervised insight into review themes.
- **Cross-Platform Comparison:** Incorporate data from VRBO, Booking.com, and local hotel APIs to benchmark Airbnb performance within the broader hospitality ecosystem.
- **Dynamic Pricing Algorithms:** Integrate demand signals (events calendar, seasonality, competitive pricing) into machine-learning–driven revenue management systems.
- **Qualitative Host Surveys:**  Complement quantitative modeling with host interviews or surveys to capture psychological/social drivers behind pricing decisions. 

## Conclusion 

This analysis transformed Airbnb listings and review data into a cohesive decision-support framework. Through systematic cleansing and enrichment, nuanced feature generation, and rigorous modeling, the project distilled complex patterns into clear, data-backed strategies. High-fidelity price forecasts and sentiment insights revealed which amenities, locations, and messaging tactics truly move the needle. By bridging advanced analytics with tangible recommendations, this work delivers a replicable playbook for optimizing revenue and guest experience—and sets the stage for scalable enhancements in dynamic pricing and feedback-driven service design.

