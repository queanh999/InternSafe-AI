const $ = (id) => document.getElementById(id);

const form = $("jobForm");
const resultSection = $("resultSection");
const errorBox = $("errorBox");
const analyzeBtn = $("analyzeBtn");
const buttonText = $("buttonText");
const buttonLoader = $("buttonLoader");

const value = (id) => $(id).value.trim();
const binary = (id) => ($(id).checked ? 1 : 0);

function setLoading(isLoading){
  analyzeBtn.disabled = isLoading;
  buttonText.textContent = isLoading ? "AI đang phân tích..." : "Phân tích bằng InternSafe AI";
  buttonLoader.classList.toggle("hidden", !isLoading);
}

function renderList(id, items, emptyText){
  const el = $(id);
  el.innerHTML = "";
  const data = items && items.length ? items : [emptyText];
  data.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    el.appendChild(li);
  });
}

function fillSample(){
  $("title").value = "Data Entry Assistant";
  $("company_profile").value = "";
  $("description").value = "Work from home. We are looking for data entry staff. Apply using the link below.";
  $("requirements").value = "";
  $("benefits").value = "";
  $("location").value = "US, NY, New York";
  $("department").value = "";
  $("salary_range").value = "";
  $("employment_type").value = "Part-time";
  $("required_experience").value = "Not Applicable";
  $("required_education").value = "High School or equivalent";
  $("industry").value = "Accounting";
  $("function").value = "Administrative";
  $("telecommuting").checked = true;
  $("has_company_logo").checked = false;
  $("has_questions").checked = false;
  updateDescriptionCount();
  document.querySelector("#scanner").scrollIntoView({behavior:"smooth"});
}

function resetForm(){
  form.reset();
  resultSection.classList.add("hidden");
  errorBox.classList.add("hidden");
  updateDescriptionCount();
}

function updateDescriptionCount(){
  $("descriptionCount").textContent = `${$("description").value.length} ký tự`;
}

$("description").addEventListener("input", updateDescriptionCount);
$("loadSample").addEventListener("click", fillSample);
$("loadSampleTop").addEventListener("click", fillSample);
$("resetForm").addEventListener("click", resetForm);

function showResult(analysis){
  const riskPanel = $("riskPanel");
  const riskBadge = $("riskBadge");
  const gauge = $("riskGauge");
  const thresholdPin = $("thresholdPin");

  riskPanel.classList.remove("risk-low","risk-review","risk-high");

  let gaugeColor = "#4f8cff";
  if(analysis.risk_level === "HIGH"){
    riskPanel.classList.add("risk-high");
    riskBadge.textContent = "HIGH RISK";
    gaugeColor = "#ff5f73";
  }else if(analysis.risk_level === "REVIEW"){
    riskPanel.classList.add("risk-review");
    riskBadge.textContent = "NEEDS REVIEW";
    gaugeColor = "#ffb84d";
  }else{
    riskPanel.classList.add("risk-low");
    riskBadge.textContent = "LOW RISK";
    gaugeColor = "#38d39f";
  }

  $("riskLabel").textContent = analysis.risk_label;
  $("riskPercent").textContent = `${analysis.risk_percent}%`;
  $("riskMessage").textContent = analysis.message;
  $("disclaimer").textContent = analysis.disclaimer;

  const p = Math.max(0, Math.min(100, Number(analysis.risk_percent) || 0));
  gauge.style.setProperty("--p", p);
  gauge.style.setProperty("--gauge-color", gaugeColor);
  thresholdPin.style.left = `${p}%`;

  renderList("riskFactors", analysis.risk_factors, "Chưa phát hiện tín hiệu cảnh báo nổi bật.");
  renderList("safetyFactors", analysis.safety_factors, "Chưa có tín hiệu tích cực nổi bật.");

  resultSection.classList.remove("hidden");
  setTimeout(() => resultSection.scrollIntoView({behavior:"smooth", block:"start"}), 120);
}

// ============================================================
// DEMO CASES - LOW / REVIEW / HIGH
// ============================================================

let demoCases = null;


// ------------------------------------------------------------
// GET DEMO CASES FROM FASTAPI
// ------------------------------------------------------------

async function getDemoCases() {

    if (demoCases) {
        return demoCases;
    }


    const response = await fetch(
        "/api/demo-cases"
    );


    if (!response.ok) {

        throw new Error(
            "Không thể tải dữ liệu demo."
        );
    }


    const result = await response.json();


    if (!result.success) {

        throw new Error(
            "Dữ liệu demo không hợp lệ."
        );
    }


    demoCases = result.cases;

    return demoCases;
}


// ------------------------------------------------------------
// SET FIELD VALUE
// ------------------------------------------------------------

function setInputValue(
    id,
    value
) {

    const element =
        document.getElementById(id);


    if (!element) {
        return;
    }


    if (
        element.type === "checkbox"
    ) {

        element.checked =
            Number(value) === 1;

    } else {

        element.value =
            value ?? "";
    }
}


// ------------------------------------------------------------
// LOAD ONE DEMO CASE
// ------------------------------------------------------------

async function loadDemoCase(
    level
) {

    errorBox.classList.add(
        "hidden"
    );


    try {

        const cases =
            await getDemoCases();


        const selected =
            cases[level];


        if (!selected) {

            throw new Error(
                "Không tìm thấy mẫu demo."
            );
        }


        const job =
            selected.job;


        const fields = [

            "title",

            "company_profile",

            "description",

            "requirements",

            "benefits",

            "location",

            "department",

            "salary_range",

            "employment_type",

            "required_experience",

            "required_education",

            "industry",

            "function",

            "telecommuting",

            "has_company_logo",

            "has_questions"

        ];


        fields.forEach(
            field => {

                setInputValue(
                    field,
                    job[field]
                );

            }
        );


        // Update character counter
        if (
            typeof updateDescriptionCount
            === "function"
        ) {

            updateDescriptionCount();
        }


        // Scroll to form
        document
            .getElementById(
                "jobForm"
            )
            .scrollIntoView({

                behavior:
                    "smooth",

                block:
                    "start"
            });


        // Automatically analyze after loading
        setTimeout(
            () => {

                form.requestSubmit();

            },
            500
        );


    } catch (error) {

        console.error(
            error
        );


        errorBox.textContent =
            "Lỗi demo: "
            + error.message;


        errorBox.classList.remove(
            "hidden"
        );
    }
}


// ------------------------------------------------------------
// CONNECT DEMO BUTTONS
// ------------------------------------------------------------

document
    .querySelectorAll(
        "[data-demo]"
    )
    .forEach(
        button => {

            button.addEventListener(
                "click",

                () => {

                    loadDemoCase(
                        button.dataset.demo
                    );

                }
            );

        }
    );

    // ============================================================
// ANALYSIS HISTORY
// ============================================================

const HISTORY_KEY =
    "internsafe_analysis_history";

function getHistory() {

    try {

        return JSON.parse(
            localStorage.getItem(
                HISTORY_KEY
            )
        ) || [];

    } catch {

        return [];
    }
}


function saveHistory(history) {

    localStorage.setItem(
        HISTORY_KEY,
        JSON.stringify(history)
    );
}


function addToHistory(
    title,
    analysis
) {

    const history =
        getHistory();

    history.unshift({

        title:
            title || "Tin tuyển dụng",

        risk_percent:
            analysis.risk_percent,

        risk_level:
            analysis.risk_level,

        time:
            new Date().toLocaleString(
                "vi-VN"
            )
    });

    saveHistory(
        history.slice(
            0,
            20
        )
    );

    renderHistory();
}


function renderHistory() {

    const history =
        getHistory();

    const list =
        document.getElementById(
            "historyList"
        );

    if (!list) {
        return;
    }


    const low =
        history.filter(
            item =>
                item.risk_level === "LOW"
        ).length;

    const review =
        history.filter(
            item =>
                item.risk_level === "REVIEW"
        ).length;

    const high =
        history.filter(
            item =>
                item.risk_level === "HIGH"
        ).length;


    document.getElementById(
        "totalAnalyses"
    ).textContent =
        history.length;

    document.getElementById(
        "lowCount"
    ).textContent =
        low;

    document.getElementById(
        "reviewCount"
    ).textContent =
        review;

    document.getElementById(
        "highCount"
    ).textContent =
        high;


    if (
        history.length === 0
    ) {

        list.innerHTML = `
            <div class="history-empty">
                Chưa có lần phân tích nào.
            </div>
        `;

        return;
    }


    list.innerHTML =
        history.map(
            item => {

                let label =
                    "RỦI RO THẤP";

                let cssClass =
                    "low";


                if (
                    item.risk_level
                    === "REVIEW"
                ) {

                    label =
                        "CẦN KIỂM TRA";

                    cssClass =
                        "review";
                }


                if (
                    item.risk_level
                    === "HIGH"
                ) {

                    label =
                        "RỦI RO CAO";

                    cssClass =
                        "high";
                }


                return `
                    <div class="history-item">

                        <div>
                            <div class="history-title">
                                ${item.title}
                            </div>

                            <span class="history-time">
                                ${item.time}
                            </span>
                        </div>

                        <div class="history-score">
                            ${item.risk_percent}%
                        </div>

                        <span
                            class="history-badge ${cssClass}"
                        >
                            ${label}
                        </span>

                    </div>
                `;

            }
        )
        .join("");
}


document
    .getElementById(
        "clearHistory"
    )
    ?.addEventListener(
        "click",
        () => {

            localStorage.removeItem(
                HISTORY_KEY
            );

            renderHistory();
        }
    );


renderHistory();
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.classList.add("hidden");

  const data = {
    title: value("title"),
    company_profile: value("company_profile"),
    description: value("description"),
    requirements: value("requirements"),
    benefits: value("benefits"),
    location: value("location"),
    department: value("department"),
    salary_range: value("salary_range"),
    employment_type: value("employment_type"),
    required_experience: value("required_experience"),
    required_education: value("required_education"),
    industry: value("industry"),
    function: value("function"),
    telecommuting: binary("telecommuting"),
    has_company_logo: binary("has_company_logo"),
    has_questions: binary("has_questions")
  };

  if(!data.title){
    errorBox.textContent = "Vui lòng nhập tiêu đề công việc.";
    errorBox.classList.remove("hidden");
    return;
  }

  setLoading(true);

  try{
    const response = await fetch("/api/analyze",{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify(data)
    });

    const result = await response.json();

    if(!response.ok){
      throw new Error(result.detail || "Không thể phân tích tin tuyển dụng.");
    }
    if(!result.success){
      throw new Error("Hệ thống không trả về kết quả hợp lệ.");
    }

    showResult(result.analysis);
    addToHistory(
    data.title,
    result.analysis
);
  }catch(err){
    console.error(err);
    errorBox.textContent = `Đã xảy ra lỗi: ${err.message}`;
    errorBox.classList.remove("hidden");
  }finally{
    setLoading(false);
  }
});




