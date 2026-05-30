┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐
│  Raw Traces  │───▸│   Parsing    │───▸│  Feature Extraction  │
│ (JSON/CSV)   │    │              │    │                      │
└──────────────┘    └──────────────┘    └──────────────────────┘
         │                  │                      │
    log/traces_raw/   processed/              processed/
    raw_traces.json   traces_parsed.csv        features_vectors.parquet
                                                    │
                                                    ▼
                                            ┌──────────────────────┐
                                            │Discipline Computation│
                                            │                      │
                                            └──────────────────────┘
                                                    │
                                              processed/
                                              discipline_metrics.csv
                                                    │
                                                    ▼
                                            ┌──────────────────────┐
                                            │    Aggregation       │
                                            │                      │
                                            └──────────────────────┘
                                                    │
                                              output/
                                              discipline_summary.csv
                                                    │
                                                    ▼
                                            ┌──────────────────────┐
                                            │    Validation        │
                                            │                      │
                                            └──────────────────────┘
                                                    │
                                              reports/
                                              validation_report.json