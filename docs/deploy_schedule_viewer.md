# 7DT Schedule viewer: deployment on the public server (7ds.snu.ac.kr)

조회 페이지(`plan_app.py`)를 7ds.snu.ac.kr 서버에서 띄우기 위한 안내입니다. 이 페이지는 읽기 전용이며, PI가 Project ID와 key를 넣으면 자기 과제의 스케줄만 보고, admin key를 넣으면 전체가 보입니다. 운영자가 lyman에서 편집·저장하는 페이지(`schedule_app.py`)는 공개 서버에서 실행하지 않습니다.

## 1. 전제

- 저장소: https://github.com/hhchoi1022/7DT_calculator (공개). 기존 페이지들처럼 `git pull`로 받습니다.
- 이 서버에 `/lyman/data1/7dt`가 NFS로 같은 경로에 마운트되어 있어야 합니다. 조회 페이지는 다음을 읽습니다 (쓰지 않음).
  - `/lyman/data1/7dt/opentime/2612/` : Phase-2 요청 파일(`*.txt`), 운영자가 저장한 `schedule.json`, `backup/`
  - `/lyman/data1/7dt/configuration/` : TCSpy 설정(관측소 위치 등), 기존 페이지들과 동일
- conda 환경은 기존 페이지들과 같습니다 (`requirements.txt`, Streamlit 1.64). 실행 시 항상 `PYTHONNOUSERSITE=1 MPLBACKEND=Agg` (run 스크립트가 넣어 줍니다).

## 2. 설치

```bash
cd /home/dtak/7DT_calculator        # 저장소 위치
git pull

# 과제 키 파일: lyman의 data/projects.json 을 그대로 복사 (git 제외 파일, 비밀)
#   lyman: /home/hhchoi1022/code/7DT_calculator_v2/data/projects.json
scp hhchoi1022@lyman:/home/hhchoi1022/code/7DT_calculator_v2/data/projects.json data/projects.json
chmod 600 data/projects.json

# 읽기 확인
ls /lyman/data1/7dt/opentime/2612/schedule.json
PYTHONNOUSERSITE=1 MPLBACKEND=Agg ~/anaconda3/envs/7dtcalc/bin/python -c "from sevendt_calc import phase2; d=phase2.requests_dir(); print(d, len(list(d.glob('*.txt'))), (d/'schedule.json').is_file())"
```

`data/settings.json`의 `phase2_requests_dir`가 이미 `/lyman/data1/7dt/opentime/2612`로 되어 있으므로 설정 변경은 필요 없습니다. 마운트 경로가 다르면 `local.env`에 `export SEVENDT_PHASE2_DIR=/다른/경로/2612`를 적으면 됩니다.

## 3. 실행

하위 경로 `/schedule/`로 서비스하므로 base path를 줍니다. 포트는 내부용으로 아무거나(예: 8520) 쓰고 방화벽에 열지 않습니다.

```bash
PORT=8520 nohup ./run_plan.sh --server.baseUrlPath schedule > ~/plan_app.log 2>&1 &
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8520/schedule/     # 200 이면 정상
```

재시작할 때:

```bash
OLD=$(pgrep -f "[s]treamlit run plan_app.py --server.port 8520" || true); [ -n "$OLD" ] && kill $OLD; sleep 2
PORT=8520 nohup ./run_plan.sh --server.baseUrlPath schedule > ~/plan_app.log 2>&1 &
```

## 4. nginx

기존 페이지들의 location 블록과 같은 방식입니다. 443 server 블록 안에 추가합니다.

```nginx
# http 블록에 한 번만: 키 추측을 느리게 (IP당 분당 30 요청)
limit_req_zone $binary_remote_addr zone=sched:10m rate=30r/m;

location /schedule/ {
    limit_req zone=sched burst=20 nodelay;
    proxy_pass         http://127.0.0.1:8520/schedule/;
    proxy_http_version 1.1;
    proxy_set_header   Upgrade $http_upgrade;
    proxy_set_header   Connection "upgrade";
    proxy_set_header   Host $host;
    proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header   X-Forwarded-Proto https;
    proxy_read_timeout 3600;
}
```

```bash
sudo nginx -t && sudo systemctl reload nginx
```

이후 https://7ds.snu.ac.kr/schedule/ 로 접속됩니다.

## 5. 확인

- 페이지가 뜨고 Project ID / Project key 입력란이 보이는지.
- 아무 과제의 ID와 key로 그 과제의 계획 표와 visibility 그림이 보이는지 (키는 `data/projects.json`).
- 상단에 "schedule as saved by the operator on …" 시각이 lyman에서 마지막으로 저장한 시각과 같은지. 운영자가 lyman에서 Save 하면 새로고침만으로 반영됩니다 (서버 재시작 불필요).

## 6. 운영 메모

- `data/projects.json`에는 과제 키와 `_admin_key`가 있습니다. 저장소에 올리지 말고 권한 600을 유지하세요. lyman에서 키를 바꾸면 이 서버에도 다시 복사해야 합니다.
- Google 서비스 계정 키는 조회 페이지에 필요 없습니다.
- 운영자 페이지(`schedule_app.py`, `run_schedule.sh`)와 Observation request 페이지는 lyman 내부 전용으로 운영합니다. 이 서버에서 실행 중이라면 내려 주세요.
- 문제 시 로그: `~/plan_app.log`.
