-- Combined Script: processing_and_complex_analysis_v3.pig

-- ==========================================================
-- Part 1: Load Raw Data and Initial Processing
-- ==========================================================

-- Load your specific CSV data.
raw_tweets = LOAD '/user/maria_dev/project/sampled_tweets.csv' USING PigStorage(',') AS (
    target:int,
    tweet_id:long,
    timestamp_str:chararray,
    flag:chararray,
    user_name:chararray,
    text:chararray
);

-- Select relevant fields and clean text
processed_data = FOREACH raw_tweets GENERATE
    target, -- Keep target field
    tweet_id,
    timestamp_str,
    user_name,
    REPLACE(REPLACE(text, '\\n', ' '), '\\t', ' ') AS cleaned_text; -- Remove newlines and tabs


-- ==========================================================
-- Part 2: Analysis
-- ==========================================================

-- 2.A Analysis: Top 10 Most Active Users
filtered_users = FILTER processed_data BY user_name IS NOT NULL;
users_grouped = GROUP filtered_users BY user_name;
user_counts = FOREACH users_grouped GENERATE group AS user_name, COUNT(filtered_users) AS tweet_count;
ordered_users = ORDER user_counts BY tweet_count DESC;
top_10_users = LIMIT ordered_users 10;
STORE top_10_users INTO '/user/maria_dev/project/sample_analysis_results/top_users' USING PigStorage(',');


-- 2.B Analysis: Top 10 Longest Tweets
filtered_text_length = FILTER processed_data BY cleaned_text IS NOT NULL;
tweet_lengths = FOREACH filtered_text_length GENERATE tweet_id, cleaned_text, SIZE(cleaned_text) as text_length;
ordered_lengths = ORDER tweet_lengths BY text_length DESC;
top_10_longest = LIMIT ordered_lengths 10;
STORE top_10_longest INTO '/user/maria_dev/project/sample_analysis_results/longest_tweets' USING PigStorage(',');


-- 2.C Analysis: Sentiment Distribution
sentiment_filtered = FILTER processed_data BY (target==0 OR target==4);
sentiment_grouped = GROUP sentiment_filtered BY target;
sentiment_counts = FOREACH sentiment_grouped GENERATE group as sentiment_label, COUNT(sentiment_filtered) as count;
STORE sentiment_counts INTO '/user/maria_dev/project/sample_analysis_results/sentiment_dist' USING PigStorage(',');


-- 2.D Analysis: Average Tweet Length
filtered_text_avg = FILTER processed_data BY cleaned_text IS NOT NULL;
lengths_only = FOREACH filtered_text_avg GENERATE SIZE(cleaned_text) as len;
grouped_all = GROUP lengths_only ALL;
avg_length_calc = FOREACH grouped_all GENERATE AVG(lengths_only.len) as avg_len;
STORE avg_length_calc INTO '/user/maria_dev/project/sample_analysis_results/avg_length' USING PigStorage(',');


-- 2.E Analysis: Top 50 Words (Basic Word Count)
words_tokenized = FOREACH processed_data GENERATE FLATTEN(TOKENIZE(LOWER(cleaned_text), ' \\t\\n\\r\\f.,!?;:()`~*#@&[]{}<>-_=+|\\/')) as word;
-- Filter out very short words
-- **** CORRECTED LINE: Used SIZE() instead of STRLEN() ****
filtered_words = FILTER words_tokenized BY SIZE(word) > 2;
words_grouped = GROUP filtered_words BY word;
word_counts = FOREACH words_grouped GENERATE group AS word, COUNT(filtered_words) as freq;
ordered_words = ORDER word_counts BY freq DESC;
top_50_words = LIMIT ordered_words 50;
STORE top_50_words INTO '/user/maria_dev/project/sample_analysis_results/top_words' USING PigStorage(',');