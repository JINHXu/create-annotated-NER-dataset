# NER on reddit posts/comments with Label Studio and SpaCy

### Get Reddit Data with Google BigQuery

subreddits: https://www.reddit.com/r/ListOfSubreddits/wiki/listofsubreddits <br>

Google BigQuery: https://console.cloud.google.com/bigquery/

Fetch reddit posts ('drugs'):
```
SELECT selftext
FROM `fh-bigquery.reddit_posts.2019_08` 
WHERE subreddit  = 'Drugs'
AND selftext != ""
AND selftext != "[removed]"
AND selftext != "[deleted]"
LIMIT 10000
```

Fectch reddit comments ('drugs'):
```
SELECT body
FROM `fh-bigquery.reddit_comments.2019_08` 
WHERE subreddit  = 'Drugs'
AND body != ""
AND body != "[removed]"
AND body != "[deleted]"
LIMIT 10000
```

### NER on unannotated data with SpaCy

* `create_trn_data.py` creates training data from reddit posts/comments with SpaCy `Matcher`
* `trn.py` runs a training loop updates a SpaCy blank model.

### Annotate Reddit Data with [Label Studio](https://labelstud.io/)

Semi-manual labeling training data:

1. Create labeling tasks for Label Studio with `create_tasks.py`
2. Create a Label Studio project `label-studio start ner_project --init`<br>
set up config
```
<View>
  <Labels name="label" toName="text">
    <Label value="DRUG" background="green"/>
  </Labels>

  <Text name="text" value="$reddit"/>
</View>
```
3. Create a spaCy backend `model.py` and connect it to the server `label-studio-ml init spacy_backend --script /path/to/model.py`<br>
`label-studio-ml start spacy_backend`<br>
*One might consider turning off VPN in case of any connection error that makes one's entire week miserable.*
4. Start Front-end `label-studio start ner_project --ml-backends http://localhost:9090`
5. Start Labeling!

