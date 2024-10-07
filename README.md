Election Info: An Information Retrieval System for U.S. 2024 Election

graph TD;
  
  %% User interaction
  User[User] --> |Submit Query| FastAPI[FastAPI Server];

  %% Query Processing
  FastAPI --> |Receive Query| Preprocessing[Query Preprocessing];
  Preprocessing --> CleanQuery[Convert to lowercase, remove special characters, remove digits];
  CleanQuery --> TokenizeQuery[Tokenize Query];
  TokenizeQuery --> LemmatizeQuery[Lemmatize Query];

  %% Query TF-IDF Computation
  LemmatizeQuery --> TermFreqQuery[Calculate Term Frequency for Query];
  TermFreqQuery --> IDFQuery[Calculate Inverse Document Frequency];
  IDFQuery --> TFIDFQuery[Compute TF-IDF Vector for Query];

  %% Document Indexing
  FastAPI --> |Load Files| LoadFiles[Load HTML Files];
  LoadFiles --> PreprocessFiles[Preprocess Documents];
  PreprocessFiles --> CleanFiles[Convert to lowercase, remove special characters, remove digits];
  CleanFiles --> TokenizeFiles[Tokenize Documents];
  TokenizeFiles --> LemmatizeFiles[Lemmatize Documents];

  %% Document TF-IDF Computation
  LemmatizeFiles --> TermFreqDocs[Calculate Term Frequency for Documents];
  TermFreqDocs --> IDFDocs[Calculate Inverse Document Frequency];
  IDFDocs --> TFIDFDocs[Compute TF-IDF Vectors for Documents];

  %% Title and Body Extraction
  LoadFiles --> ExtractContent[Extract Title and Body];
  ExtractContent --> StoreContent[Store Title and Body];

  %% Document Storage
  StoreContent --> DocumentStorage[Document Storage];

  %% Cosine Similarity Calculation
  TFIDFQuery --> CosineSim[Compute Cosine Similarity];
  TFIDFDocs --> CosineSim;
  
  CosineSim --> RankDocs[Rank Documents by Relevance];
  
  %% Response Handling
  RankDocs --> |Return Ranked Results| Response[Generate HTML Response];
  Response --> User[User];
  
  %% Meta Description Extraction
  LoadFiles --> ExtractMeta[Extract Meta Descriptions and Title];
  ExtractMeta --> MetaDescriptions[Store Meta Descriptions and Title];

  %% Display elements
  style User fill:#f9f,stroke:#333,stroke-width:2px;
  style FastAPI fill:#f96,stroke:#333,stroke-width:2px;
  style Preprocessing fill:#bbf,stroke:#333,stroke-width:2px;
  style PreprocessFiles fill:#bbf,stroke:#333,stroke-width:2px;
  style CosineSim fill:#bbf,stroke:#333,stroke-width:2px;
  style RankDocs fill:#bbf,stroke:#333,stroke-width:2px;
  style DocumentStorage fill:#ff6,stroke:#333,stroke-width:2px;
  style ExtractMeta fill:#6f9,stroke:#333,stroke-width:2px;
  style MetaDescriptions fill:#6f9,stroke:#333,stroke-width:2px;
  style Response fill:#f96,stroke:#333,stroke-width:2px;
  style ExtractContent fill:#6f9,stroke:#333,stroke-width:2px;
  style StoreContent fill:#6f9,stroke:#333,stroke-width:2px;
