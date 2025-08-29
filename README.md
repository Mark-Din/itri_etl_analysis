Overall Info

## The repository is for testing the backup and python ETL process from PostgreSQL to MySQL of the Concord system.


## python_etl_container_compose container 
### Path
/opt/penta_QR/python
### Files needed
code  cron_exe_docker.sh  crontab  Dockerfile_python  logs  __pycache__  python-image.tar  startup.sh

## quickreport_app_container_compose container
### Path
/opt/penta_QR/docker_mongo_qr
### Files needed
data  Dockerfile_mongo_qr  entrypoint.sh  logs  mongodb-reporting_newest.tar  quickreport-app-image.tar  report  reporting  report_server_docker.sh  report.zip

## Cron in python_etl_container_compose container 
*/5 * * * * /usr/local/bin/python3 /app/main.py >> /app/cronS_log/cron.log 2>&1

0 0 * * * /usr/local/bin/python3 /app/deleting_db_logs.py >> /app/cron_log/cron.log 2>&1

0 0 * * * /usr/local/bin/python3 /app/log_manager.py >> /app/cron_log/rotate.log 2>&1

========================================================================================

## backup and delete process
### Path
/opt/Concord_backup_delete_process
### Files needed
backup_sql_concord_smb.sh  config.sh  config.sh.enc  encrypt_pass.sh

## Cron
4,16 * * * * cd /opt/penta_db_backup && ./backup_sql_concord_smb.sh >> /var/lib/backupFile/procedure_log.log 2>&1

10 4 * * * cd /opt/penta_db_backup && ./deleting_backup.sh >> /var/lib/backupFile/procedure_log.log 2>&1

