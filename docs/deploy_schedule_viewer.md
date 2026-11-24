# 7DT Schedule viewer: deployment on the public server (7ds.snu.ac.kr)

조회 페이지(`plan_app.py`)를 7ds.snu.ac.kr 서버에서 띄우기 위한 안내입니다. 이 페이지는 읽기 전용이며, PI가 Project ID와 key를 넣으면 자기 과제의 스케줄만 보고, admin key를 넣으면 전체가 보입니다. 운영자가 lyman에서 편집·저장하는 페이지(`schedule_app.py`)는 공개 서버에서 실행하지 않습니다.

## 1. 전제

- 저장소: https://github.com/hhchoi1022/7DT_calculator (공개). 기존 페이지들처럼 `git pull`로 받습니다.
- 이 서버에 `/lyman/data1/7dt`가 NFS로 같은 경로에 마운트되어 있어야 합니다. 조회 페이지는 다음을 읽습니다 (쓰지 않음).
  - `/lyman/data1/7dt/opentime/2612/` : Phase-2 요청 파일(`*.txt`), 운영자가 저장한 `schedule.json`, `backup/`
  - `/lyman/data1/7dt/configuration/` : TCSpy 설정(관측소 위치 등), 기존 페이지들과 동일
- conda 환경은 기존 페이지들과 같습니다 (`requirements.txt`, Streamlit 1.64). 실행 시 항상 `PYTHONNOUSERSITE=1 MPLBACKEND=Agg` (run 스크립트가 넣어 줍니다). run 스크립트는 `~/anaconda3`와 `~/miniconda3` 중 `7dtcalc` 환경이 있는 쪽을 쓰고, `PYTHON=/경로/python`으로 지정할 수도 있습니다. 조회 페이지는 `gspread`, `google-auth`가 필요 없으니 새로 설치할 패키지는 없습니다.
- `data/projects.json`이 없으면 조회 페이지는 아무것도 보여 주지 않습니다 (키 검사 없이 열리지 않도록 막아 두었습니다). 2절의 복사 단계는 반드시 필요합니다.

## 2. 설치

```bash
cd /home/dtak/7DT_calculator        # 저장소 위치
git pull
grep -rn '^<<<<<<<' --include='*.py' . ; git status --short     # 충돌 흔적이 없어야 합니다 (머지 커밋이 생기는 것은 정상)

# 과제 키 파일: lyman의 data/projects.json 을 그대로 복사 (git 제외 파일, 비밀)
#   lyman: /home/hhchoi1022/code/7DT_calculator_v2/data/projects.json
scp hhchoi1022@lyman:/home/hhchoi1022/code/7DT_calculator_v2/data/projects.json data/projects.json
chmod 600 data/projects.json

# 읽기 확인
ls /lyman/data1/7dt/opentime/2612/schedule.json
PY=$(ls -d ~/anaconda3/envs/7dtcalc/bin/python ~/miniconda3/envs/7dtcalc/bin/python 2>/dev/null | head -1)
PYTHONNOUSERSITE=1 MPLBACKEND=Agg $PY -c "from sevendt_calc import phase2; d=phase2.requests_dir(); print(d, len(list(d.glob('*.txt'))), (d/'schedule.json').is_file())"
```

`git pull`은 이미 떠 있는 페이지들이 쓰는 파일(`ui_common.py`, `sevendt_calc/`)도 바꿉니다. 떠 있는 Streamlit 프로세스에는 옛 모듈이 남아 있어 재시작 전까지 오류가 날 수 있으니, pull 뒤에는 기존 페이지들(visibility, exptime, overhead, tile)도 같은 방법으로 재시작해 주세요.

`data/settings.json`의 `phase2_requests_dir`가 이미 `/lyman/data1/7dt/opentime/2612`로 되어 있으므로 설정 변경은 필요 없습니다. 마운트 경로가 다르면 `local.env`에 `export SEVENDT_PHASE2_DIR=/다른/경로/2612`를 적으면 됩니다.

## 3. 실행

하위 경로 `/schedule/`로 서비스하므로 base path를 줍니다. 포트는 내부용으로 아무거나(예: 8520) 쓰고, 기존 페이지들처럼 `--server.address 127.0.0.1`을 붙여 nginx를 통해서만 닿게 합니다 (방화벽에 열지 않습니다).

```bash
PORT=8520 nohup ./run_plan.sh --server.address 127.0.0.1 --server.baseUrlPath schedule > ~/plan_app.log 2>&1 &
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8520/schedule/     # 200 이면 정상
```

재시작할 때:

```bash
OLD=$(pgrep -f "[s]treamlit run plan_app.py --server.port 8520" || true); [ -n "$OLD" ] && kill $OLD; sleep 2
PORT=8520 nohup ./run_plan.sh --server.address 127.0.0.1 --server.baseUrlPath schedule > ~/plan_app.log 2>&1 &
```

## 4. nginx

기존 페이지들의 location 블록과 같은 방식입니다. 443 server 블록 안에 추가합니다. 요청 횟수 제한(`limit_req`)은 넣지 않습니다: 키는 websocket 안으로 오가므로 HTTP 요청 제한으로는 추측을 늦추지 못하고, 첫 화면 로딩(정적 파일 20개 안팎)만 503으로 끊길 수 있습니다. 키 추측은 페이지 자체가 막습니다 (IP와 Project ID마다 틀린 키 8번까지는 바로, 그 뒤로는 10초부터 두 배씩 늘어나는 대기, `sevendt_calc/keyguard.py`). `X-Forwarded-For` 헤더는 그 IP 구분에 쓰이므로 꼭 넘겨 주세요.

```nginx
location /schedule/ {
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
- 아무 과제의 ID와 key로 그 과제의 계획 표와 visibility 그림이 보이는지 (키는 `data/projects.json`). 틀린 키에는 "The Project key does not match." 가 떠야 합니다. "The project list (data/projects.json) is not configured on this server" 가 뜨면 2절의 복사가 안 된 것입니다.
- 상단에 "schedule as saved by the operator on …" 시각이 lyman에서 마지막으로 저장한 시각과 같은지. 운영자가 lyman에서 Save 하면 새로고침만으로 반영됩니다 (서버 재시작 불필요).

## 6. 운영 메모

- `data/projects.json`에는 과제 키와 `_admin_key`가 있습니다. 저장소에 올리지 말고 권한 600을 유지하세요. lyman에서 키를 바꾸면 이 서버에도 다시 복사해야 합니다. 파일이 없으면 페이지는 아무 스케줄도 보여 주지 않습니다.
- 틀린 키 대기는 프로세스 메모리에만 있으므로 페이지를 재시작하면 초기화됩니다. 한 시간 지난 실패는 잊습니다.
- Google 서비스 계정 키는 조회 페이지에 필요 없습니다.
- 운영자 페이지(`schedule_app.py`, `run_schedule.sh`)와 Observation request 페이지는 lyman 내부 전용으로 운영합니다. 이 서버에서 실행 중이라면 내려 주세요.
- 문제 시 로그: `~/plan_app.log`.
