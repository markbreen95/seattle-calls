SELECT EXTRACT(YEAR FROM report_date) AS year, EXTRACT(MONTH FROM report_date) AS month, EXTRACT(DAY FROM report_date) AS day, CAST(SUBSTR(report_time, 0, 2) AS INT) AS hour, Longitude, Latitude FROM `niologic-assessment.seattle_calls.calls-processed` WHERE EXTRACT(YEAR FROM report_date) > 2021;