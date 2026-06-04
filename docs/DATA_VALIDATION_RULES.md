{
  "pipeline_run_id": "string (UUID)",
  "timestamp": "ISO 8601 datetime",
  "total_records_checked": "integer",
  "passed": "integer",
  "failed": "integer",
  "soft_failures": "integer",
  "rules_applied": ["list of rule IDs"],
  "failures": [
    {
      "rule_id": "string",
      "record_index": "integer (0-indexed)",
      "field": "string",
      "value": "any",
      "message": "string"
    }
  ],
  "soft_failures_list": [
    {
      "rule_id": "string",
      "record_index": "integer",
      "field": "string",
      "value": "any",
      "message": "string"
    }
  ],
  "summary": {
    "percent_pass": "float",
    "critical_errors": "list of strings"
  }
}