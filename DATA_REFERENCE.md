# SentinelRisk Data Reference

## raw_events Table Structure

### Columns

| Column | Data Type |
|--------|-----------|
| event_id | text |
| ts | timestamp with time zone |
| entity_type | USER-DEFINED |
| entity_id | text |
| event_type | text |
| status | USER-DEFINED |
| amount | numeric |
| currency | text |
| user_id | text |
| merchant_id | text |
| api_client_id | text |
| device_id | text |
| ip_hash | text |
| ua_hash | text |
| endpoint | text |
| error_code | text |
| channel | text |
| source_dataset | text |
| ingest_batch_id | text |
| metadata | jsonb |

---

## Sample Data by Source

### CERT

```
   entity_id entity_type event_type                        ts amount  status channel
DTAA/KEE0997        user      login 2010-01-04 00:10:37+00:00   None success    None
DTAA/KEE0997        user      login 2010-01-04 00:52:16+00:00   None success    None
DTAA/KEE0997        user      login 2010-01-04 01:17:20+00:00   None success    None
DTAA/KEE0997        user      login 2010-01-04 01:28:34+00:00   None success    None
DTAA/BJM0992        user      login 2010-01-04 01:57:30+00:00   None success    None
```

**Columns with data**: event_id, ts, entity_type, entity_id, event_type, status, user_id, device_id, source_dataset, ingest_batch_id...

**Mostly NULL**: amount, currency, merchant_id, api_client_id, ip_hash...


### IEEE_CIS

```
   entity_id entity_type       event_type                        ts  amount  status channel
user_3577237        user transaction_auth 2018-06-01 22:12:04+00:00 117.000 success payment
user_3577238        user transaction_auth 2018-06-01 22:12:05+00:00  42.950 success payment
user_3577239        user transaction_auth 2018-06-01 22:12:07+00:00  31.950 success payment
user_3577240        user transaction_auth 2018-06-01 22:12:20+00:00  29.757 success payment
user_3577241        user transaction_auth 2018-06-01 22:12:21+00:00  59.000 success payment
```

**Columns with data**: event_id, ts, entity_type, entity_id, event_type, status, amount, currency, user_id, channel...

**Mostly NULL**: merchant_id, api_client_id, device_id, ip_hash, ua_hash...


### CREDITCARD

```
entity_id entity_type       event_type                        ts  amount  status      channel
 cc_txn_0        user transaction_auth 2013-09-01 00:00:00+00:00  149.62 success card_present
 cc_txn_1        user transaction_auth 2013-09-01 00:00:00+00:00    2.69 success card_present
 cc_txn_2        user transaction_auth 2013-09-01 00:00:01+00:00  378.66 success card_present
 cc_txn_3        user transaction_auth 2013-09-01 00:00:01+00:00  123.50 success card_present
 cc_txn_4        user transaction_auth 2013-09-01 00:00:02+00:00   69.99 success card_present
```

**Columns with data**: event_id, ts, entity_type, entity_id, event_type, status, amount, currency, user_id, channel...

**Mostly NULL**: merchant_id, api_client_id, device_id, ip_hash, ua_hash...


### NAB

```
                         entity_id   entity_type         event_type                        ts    amount  status channel
ambient_temperature_system_failure system_metric metric_observation 2013-07-04 00:00:00+00:00 69.880835 success    None
ambient_temperature_system_failure system_metric metric_observation 2013-07-04 01:00:00+00:00 71.220227 success    None
ambient_temperature_system_failure system_metric metric_observation 2013-07-04 02:00:00+00:00 70.877805 success    None
ambient_temperature_system_failure system_metric metric_observation 2013-07-04 03:00:00+00:00 68.959400 success    None
ambient_temperature_system_failure system_metric metric_observation 2013-07-04 04:00:00+00:00 69.283551 success    None
```

**Columns with data**: event_id, ts, entity_type, entity_id, event_type, status, amount, source_dataset, ingest_batch_id, metadata

**Mostly NULL**: currency, user_id, merchant_id, api_client_id, device_id...

