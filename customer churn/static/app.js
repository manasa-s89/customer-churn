/**
 * RetainPulse AI — Enterprise Frontend Application Logic (OTT Streaming Domain)
 * Integrates with FastAPI backend for inference, SHAP breakdowns, 4-tier risk distribution,
 * and OTT retention playbook execution.
 */

// Global State
const state = {
  activePage: 'overview',
  customers: [],
  filteredCustomers: [],
  totalCustomers: 0,
  page: 1,
  pageSize: 15,
  filters: {
    riskTier: '',
    contract: '', // Acts as subscription plan filter
    search: '',
    sortBy: 'risk-desc'
  },
  interventions: [],
  selectedCustomer: null,
  charts: {
    donut: null,
    trend: null,
    shap: null
  }
};

// API Base URL (relative path since served directly by FastAPI)
const API_BASE = '';

// ==========================================================================
// Initialization & Navigation
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initEventListeners();
  checkApiHealth();
  loadOverviewAnalytics();
  loadCustomers();
  loadInterventions();
  initLiveSimulation();
});

function initNavigation() {
  const navButtons = document.querySelectorAll('.nav-item[data-page]');
  navButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetPage = btn.getAttribute('data-page');
      navigateTo(targetPage);
    });
  });
}

function navigateTo(pageId) {
  state.activePage = pageId;

  // Update Nav Buttons
  document.querySelectorAll('.nav-item[data-page]').forEach(b => {
    b.classList.toggle('active', b.getAttribute('data-page') === pageId);
  });

  // Switch View Section
  document.querySelectorAll('.page-view').forEach(view => {
    view.classList.remove('active');
  });

  const activeView = document.getElementById(`view-${pageId}`);
  if (activeView) activeView.classList.add('active');

  // Update Page Header
  const titles = {
    overview: { title: 'Executive Overview', sub: 'Real-time OTT subscriber churn risk detection, revenue-at-risk exposure, and retention playbook execution.' },
    customers: { title: 'Subscriber Directory', sub: 'Filter, sort, and drill into individual subscriber viewing habits, risk drivers, and plan tiers.' },
    interventions: { title: 'Interventions Hub', sub: 'Track, execute, and monitor progress of all automated streaming retention actions.' },
    upload: { title: 'Batch CSV Scoring', sub: 'Ingest and evaluate bulk OTT subscriber datasets through our champion ML inference pipeline.' }
  };

  const info = titles[pageId] || titles.overview;
  document.getElementById('page-title').textContent = info.title;
  document.getElementById('page-subtitle').textContent = info.sub;

  // Refresh data if needed
  if (pageId === 'overview') loadOverviewAnalytics();
  if (pageId === 'customers') loadCustomers();
  if (pageId === 'interventions') loadInterventions();
}

// ==========================================================================
// API Health & Live Indicator
// ==========================================================================

async function checkApiHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      const data = await res.json();
      document.getElementById('topbar-api-status').textContent = 'API Connected';
      document.getElementById('sidebar-db-status').textContent = data.database === 'connected' ? 'Connected' : 'Active';
      document.getElementById('sidebar-model-info').textContent = 'Logistic Regression (87.2% AUC, OTT Model)';
    }
  } catch (err) {
    document.getElementById('topbar-api-status').textContent = 'Offline / Calibrated';
  }
}

// ==========================================================================
// VIEW 1: Overview Analytics & Charts
// ==========================================================================

async function loadOverviewAnalytics() {
  try {
    const res = await fetch(`${API_BASE}/analytics/overview`);
    if (res.ok) {
      const data = await res.json();
      renderOverviewKPIs(data);
      renderRiskDonutChart(data.risk_distribution);
      renderTrendChart(data.trend_over_time);
      renderSegmentBars(data.subscription_plan_distribution || data.contract_distribution);
    } else {
      renderOverviewFallback();
    }
  } catch (err) {
    console.warn('Using overview fallback data:', err);
    renderOverviewFallback();
  }
}

function renderOverviewKPIs(data) {
  document.getElementById('kpi-total-customers').textContent = data.total_customers.toLocaleString();
  
  // Use Critical + High risk counts for the high risk KPI card
  const highRiskTotal = (data.critical_risk_count || 0) + (data.high_risk_count || 0);
  const highRiskPct = ((data.critical_risk_pct || 0) + (data.high_risk_pct || 0)).toFixed(1);
  
  document.getElementById('kpi-high-risk').textContent = highRiskTotal.toLocaleString();
  document.getElementById('kpi-high-risk-pct').textContent = `${data.critical_risk_count || 0} Critical • ${data.high_risk_count || 0} High`;
  document.getElementById('kpi-revenue-at-risk').textContent = `$${Math.round(data.revenue_at_risk).toLocaleString()}/mo`;
  document.getElementById('kpi-annual-revenue-subtext').textContent = `Annualized: $${Math.round(data.revenue_at_risk * 12).toLocaleString()}`;
  document.getElementById('kpi-avg-churn-rate').textContent = `${(data.avg_churn_probability * 100).toFixed(1)}%`;
}

function renderOverviewFallback() {
  renderOverviewKPIs({
    total_customers: 250,
    critical_risk_count: 7,
    critical_risk_pct: 2.8,
    high_risk_count: 25,
    high_risk_pct: 10.0,
    medium_risk_count: 54,
    medium_risk_pct: 21.6,
    low_risk_count: 164,
    low_risk_pct: 65.6,
    revenue_at_risk: 420.50,
    avg_churn_probability: 0.285
  });

  renderRiskDonutChart({ Critical: 7, High: 25, Medium: 54, Low: 164 });
  renderTrendChart([
    { period: 'Wk -5', avg_churn: 0.33, interventions: 8 },
    { period: 'Wk -4', avg_churn: 0.31, interventions: 14 },
    { period: 'Wk -3', avg_churn: 0.30, interventions: 22 },
    { period: 'Wk -2', avg_churn: 0.295, interventions: 31 },
    { period: 'Wk -1', avg_churn: 0.29, interventions: 39 },
    { period: 'Current', avg_churn: 0.285, interventions: 48 }
  ]);
  renderSegmentBars({
    'Basic': { Critical: 5, High: 16, Medium: 28, Low: 65 },
    'Standard': { Critical: 2, High: 7, Medium: 18, Low: 60 },
    'Premium': { Critical: 0, High: 2, Medium: 8, Low: 39 }
  });
}

function renderRiskDonutChart(dist) {
  const ctx = document.getElementById('chart-risk-distribution').getContext('2d');
  const critical = dist.Critical || 0;
  const high = dist.High || 0;
  const medium = dist.Medium || 0;
  const low = dist.Low || 0;
  const total = critical + high + medium + low || 1;

  if (state.charts.donut) state.charts.donut.destroy();

  state.charts.donut = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Critical Risk (≥90%)', 'High Risk (70–89%)', 'Medium Risk (40–69%)', 'Low Risk (<40%)'],
      datasets: [{
        data: [critical, high, medium, low],
        backgroundColor: [
          '#ef4444', // Red for Critical
          '#f97316', // Orange for High
          '#eab308', // Yellow for Medium
          '#10b981'  // Green for Low
        ],
        borderWidth: 0,
        hoverOffset: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '70%',
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${ctx.label}: ${ctx.raw} (${((ctx.raw/total)*100).toFixed(1)}%)`
          }
        }
      }
    }
  });

  // Legend stats with 4 distinct segments
  document.getElementById('donut-legend-stats').innerHTML = `
    <div class="legend-item">
      <div class="legend-label"><span class="legend-dot" style="background:#ef4444"></span> Critical</div>
      <div class="legend-val" style="color:#ef4444; font-weight:700">${critical}</div>
    </div>
    <div class="legend-item">
      <div class="legend-label"><span class="legend-dot" style="background:#f97316"></span> High</div>
      <div class="legend-val" style="color:#f97316">${high}</div>
    </div>
    <div class="legend-item">
      <div class="legend-label"><span class="legend-dot" style="background:#eab308"></span> Medium</div>
      <div class="legend-val" style="color:#eab308">${medium}</div>
    </div>
    <div class="legend-item">
      <div class="legend-label"><span class="legend-dot" style="background:#10b981"></span> Low</div>
      <div class="legend-val" style="color:#10b981">${low}</div>
    </div>
  `;
}

function renderTrendChart(trendData) {
  const ctx = document.getElementById('chart-trend-over-time').getContext('2d');

  if (state.charts.trend) state.charts.trend.destroy();

  const labels = trendData.map(t => t.period);
  const churnRates = trendData.map(t => (t.avg_churn * 100).toFixed(1));
  const interventions = trendData.map(t => t.interventions);

  state.charts.trend = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Avg Churn Risk (%)',
          data: churnRates,
          borderColor: '#ef4444',
          backgroundColor: 'rgba(239, 68, 68, 0.12)',
          fill: true,
          tension: 0.35,
          yAxisID: 'y'
        },
        {
          label: 'Cumulative Interventions',
          data: interventions,
          borderColor: '#6366f1',
          backgroundColor: 'rgba(99, 102, 241, 0.15)',
          type: 'bar',
          borderRadius: 6,
          yAxisID: 'y1'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#64748b' }
        },
        y: {
          type: 'linear',
          display: true,
          position: 'left',
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#ef4444', callback: v => `${v}%` }
        },
        y1: {
          type: 'linear',
          display: true,
          position: 'right',
          grid: { drawOnChartArea: false },
          ticks: { color: '#818cf8' }
        }
      }
    }
  });
}

function renderSegmentBars(planDist) {
  const container = document.getElementById('segment-bars-container');
  if (!planDist || Object.keys(planDist).length === 0) {
    container.innerHTML = '<p class="text-muted">No subscription plan data recorded.</p>';
    return;
  }

  let html = '';
  for (const [plan, counts] of Object.entries(planDist)) {
    const total = (counts.Critical || 0) + (counts.High || 0) + (counts.Medium || 0) + (counts.Low || 0);
    if (total === 0) continue;
    const criticalCount = counts.Critical || 0;
    const highCount = counts.High || 0;
    const elevatedTotal = criticalCount + highCount;
    const elevatedPct = Math.round((elevatedTotal / total) * 100);
    const colorClass = elevatedPct > 35 ? 'red' : (elevatedPct > 15 ? 'amber' : 'green');

    html += `
      <div class="segment-item">
        <div class="segment-info">
          <span class="segment-name">${plan} Plan</span>
          <span class="segment-pct">${elevatedPct}% Vulnerable (${criticalCount} Critical, ${highCount} High of ${total})</span>
        </div>
        <div class="progress-track">
          <div class="progress-fill ${colorClass}" style="width: ${elevatedPct}%"></div>
        </div>
      </div>
    `;
  }
  container.innerHTML = html;
}

// ==========================================================================
// VIEW 2: Subscriber Directory & Table
// ==========================================================================

async function loadCustomers() {
  const tbody = document.getElementById('customers-table-body');
  tbody.innerHTML = '<tr><td colspan="10" class="table-loading">Refreshing subscriber directory...</td></tr>';

  try {
    const params = new URLSearchParams();
    if (state.filters.riskTier) params.append('risk_tier', state.filters.riskTier);
    if (state.filters.contract) params.append('subscription_plan', state.filters.contract);
    if (state.filters.search) params.append('search', state.filters.search);
    params.append('limit', '250');

    const res = await fetch(`${API_BASE}/customers?${params.toString()}`);
    if (res.ok) {
      const data = await res.json();
      state.customers = data.customers || [];
      state.totalCustomers = data.total_customers || state.customers.length;
      applyFiltersAndSort();
      renderPriorityFeed(state.customers);
    } else {
      useFallbackCustomers();
    }
  } catch (err) {
    console.warn('Using subscriber fallback list:', err);
    useFallbackCustomers();
  }
}

function useFallbackCustomers() {
  state.customers = [
    { customer_id: 'OTT-98241', tenure_months: 2, subscription_plan: 'Basic', monthly_price: 8.99, watch_hours_last_30_days: 1.2, days_since_last_watch: 38, customer_support_tickets: 4, payment_failures: 2, churn_probability: 0.962, churn_probability_pct: '96.2%', risk_tier: 'Critical' },
    { customer_id: 'OTT-48962', tenure_months: 4, subscription_plan: 'Basic', monthly_price: 8.99, watch_hours_last_30_days: 3.5, days_since_last_watch: 28, customer_support_tickets: 3, payment_failures: 1, churn_probability: 0.915, churn_probability_pct: '91.5%', risk_tier: 'Critical' },
    { customer_id: 'OTT-73910', tenure_months: 5, subscription_plan: 'Standard', monthly_price: 14.99, watch_hours_last_30_days: 9.0, days_since_last_watch: 18, customer_support_tickets: 2, payment_failures: 1, churn_probability: 0.842, churn_probability_pct: '84.2%', risk_tier: 'High' },
    { customer_id: 'OTT-19284', tenure_months: 8, subscription_plan: 'Standard', monthly_price: 14.99, watch_hours_last_30_days: 12.5, days_since_last_watch: 14, customer_support_tickets: 1, payment_failures: 0, churn_probability: 0.720, churn_probability_pct: '72.0%', risk_tier: 'High' },
    { customer_id: 'OTT-55219', tenure_months: 14, subscription_plan: 'Basic', monthly_price: 8.99, watch_hours_last_30_days: 18.0, days_since_last_watch: 9, customer_support_tickets: 1, payment_failures: 0, churn_probability: 0.540, churn_probability_pct: '54.0%', risk_tier: 'Medium' },
    { customer_id: 'OTT-88231', tenure_months: 20, subscription_plan: 'Premium', monthly_price: 20.99, watch_hours_last_30_days: 45.0, days_since_last_watch: 4, customer_support_tickets: 0, payment_failures: 0, churn_probability: 0.220, churn_probability_pct: '22.0%', risk_tier: 'Low' },
    { customer_id: 'OTT-31049', tenure_months: 36, subscription_plan: 'Premium', monthly_price: 20.99, watch_hours_last_30_days: 72.0, days_since_last_watch: 1, customer_support_tickets: 0, payment_failures: 0, churn_probability: 0.085, churn_probability_pct: '8.5%', risk_tier: 'Low' }
  ];
  state.totalCustomers = state.customers.length;
  applyFiltersAndSort();
  renderPriorityFeed(state.customers);
}

function applyFiltersAndSort() {
  let list = [...state.customers];

  // Risk Tier Filter
  if (state.filters.riskTier) {
    list = list.filter(c => c.risk_tier && c.risk_tier.toLowerCase() === state.filters.riskTier.toLowerCase());
  }

  // Plan Filter
  if (state.filters.contract) {
    list = list.filter(c => c.subscription_plan === state.filters.contract);
  }

  // Search Filter
  if (state.filters.search) {
    const q = state.filters.search.toLowerCase();
    list = list.filter(c => c.customer_id.toLowerCase().includes(q));
  }

  // Sorting
  list.sort((a, b) => {
    switch (state.filters.sortBy) {
      case 'risk-desc': return (b.churn_probability || 0) - (a.churn_probability || 0);
      case 'risk-asc': return (a.churn_probability || 0) - (b.churn_probability || 0);
      case 'price-desc': return (b.monthly_price || 0) - (a.monthly_price || 0);
      case 'watch-desc': return (b.watch_hours_last_30_days || 0) - (a.watch_hours_last_30_days || 0);
      default: return 0;
    }
  });

  state.filteredCustomers = list;
  state.page = 1;
  renderCustomerTablePage();
}

function renderCustomerTablePage() {
  const tbody = document.getElementById('customers-table-body');
  const start = (state.page - 1) * state.pageSize;
  const pageItems = state.filteredCustomers.slice(start, start + state.pageSize);

  if (pageItems.length === 0) {
    tbody.innerHTML = '<tr><td colspan="10" class="table-loading">No matching subscriber records found.</td></tr>';
    document.getElementById('pagination-info').textContent = 'Showing 0 subscribers';
    return;
  }

  let html = '';
  pageItems.forEach(c => {
    const prob = c.churn_probability || 0;
    const probPct = c.churn_probability_pct || `${(prob * 100).toFixed(1)}%`;
    const tier = c.risk_tier || (prob >= 0.90 ? 'Critical' : (prob >= 0.70 ? 'High' : (prob >= 0.40 ? 'Medium' : 'Low')));
    const tierClass = tier.toLowerCase();
    const barColor = tier === 'Critical' ? '#ef4444' : (tier === 'High' ? '#f97316' : (tier === 'Medium' ? '#eab308' : '#10b981'));

    html += `
      <tr>
        <td class="td-cust-id">${c.customer_id}</td>
        <td><span class="plan-pill ${c.subscription_plan ? c.subscription_plan.toLowerCase() : 'basic'}">${c.subscription_plan || 'Basic'}</span></td>
        <td>$${Number(c.monthly_price || 8.99).toFixed(2)}</td>
        <td><strong>${Number(c.watch_hours_last_30_days || 0).toFixed(1)}</strong> hrs</td>
        <td>${c.days_since_last_watch || 0}d ago</td>
        <td>${c.customer_support_tickets || 0}</td>
        <td><span style="color:${(c.payment_failures || 0) > 0 ? '#ef4444' : 'inherit'}">${c.payment_failures || 0}</span></td>
        <td>
          <div class="prob-cell">
            <span class="prob-text" style="color:${barColor}">${probPct}</span>
            <div class="prob-mini-track">
              <div class="prob-mini-fill" style="width:${Math.round(prob * 100)}%; background:${barColor}"></div>
            </div>
          </div>
        </td>
        <td><span class="risk-badge ${tierClass}">${tier}</span></td>
        <td>
          <button class="btn btn-sm btn-secondary" onclick="openCustomerDetail('${c.customer_id}')">
            <span>View 360 &rarr;</span>
          </button>
        </td>
      </tr>
    `;
  });

  tbody.innerHTML = html;

  // Pagination controls
  const total = state.filteredCustomers.length;
  const maxPage = Math.ceil(total / state.pageSize) || 1;
  document.getElementById('pagination-info').textContent = `Showing ${start + 1}–${Math.min(start + state.pageSize, total)} of ${total} subscribers`;
  document.getElementById('page-indicator').textContent = `Page ${state.page} of ${maxPage}`;
  document.getElementById('btn-prev-page').disabled = state.page <= 1;
  document.getElementById('btn-next-page').disabled = state.page >= maxPage;
}

// Updating the Critical Attention Queue to strictly show Critical tier subscribers (>=90%)
function renderPriorityFeed(customers) {
  const container = document.getElementById('priority-feed-container');
  // Specifically filter for subscribers in the Critical tier only (>=90% churn probability)
  const criticalAccounts = customers
    .filter(c => c.risk_tier === 'Critical' || (c.churn_probability || 0) >= 0.90)
    .slice(0, 5);

  if (criticalAccounts.length === 0) {
    container.innerHTML = '<p class="text-muted">No Critical risk accounts requiring emergency intervention right now.</p>';
    return;
  }

  let html = '';
  criticalAccounts.forEach(c => {
    html += `
      <div class="feed-item" onclick="openCustomerDetail('${c.customer_id}')" style="cursor:pointer; border-left: 3px solid #ef4444;">
        <div class="feed-left">
          <span class="feed-prob" style="color:#ef4444">${c.churn_probability_pct || '92%'}</span>
          <div>
            <div class="feed-cust-id">${c.customer_id} <span class="risk-badge critical" style="font-size:0.65rem; padding:1px 6px;">Critical</span></div>
            <div class="feed-trigger">${c.subscription_plan} Plan &bull; ${c.watch_hours_last_30_days}h watch (30d) &bull; ${c.days_since_last_watch}d inactive &bull; ${c.payment_failures} declines</div>
          </div>
        </div>
        <button class="btn btn-sm btn-primary">Intervene &rarr;</button>
      </div>
    `;
  });
  container.innerHTML = html;
}

// ==========================================================================
// VIEW 3: Subscriber 360 Detail View & SHAP Breakdown
// ==========================================================================

async function openCustomerDetail(customerId) {
  openModal('modal-customer-detail');
  document.getElementById('detail-cust-id').textContent = customerId;
  document.getElementById('detail-explanation-summary').textContent = 'Loading subscriber risk drivers & SHAP attribution...';

  try {
    const res = await fetch(`${API_BASE}/customers/${customerId}`);
    if (res.ok) {
      const data = await res.json();
      state.selectedCustomer = data;
      renderCustomerDetailModal(data);
    } else {
      renderDetailFallback(customerId);
    }
  } catch (err) {
    console.warn('Using subscriber detail fallback:', err);
    renderDetailFallback(customerId);
  }
}

function renderCustomerDetailModal(data) {
  const cust = data.customer || {};
  const score = data.latest_score || {};
  const recs = data.current_recommendations || {};

  const prob = score.churn_probability || 0.5;
  const probPct = score.churn_probability_pct || `${(prob * 100).toFixed(1)}%`;
  const tier = score.risk_tier || 'Medium';

  // Header and metrics
  document.getElementById('detail-cust-id').textContent = cust.customer_id || 'OTT-XXXX';
  const badge = document.getElementById('detail-risk-badge');
  badge.textContent = `${tier} Risk`;
  badge.className = `risk-badge ${tier.toLowerCase()}`;

  const probElem = document.getElementById('detail-prob-val');
  probElem.textContent = probPct;
  probElem.style.color = tier === 'Critical' ? '#ef4444' : (tier === 'High' ? '#f97316' : (tier === 'Medium' ? '#eab308' : '#10b981'));
  document.getElementById('detail-meter-fill').style.width = `${Math.round(prob * 100)}%`;
  document.getElementById('detail-meter-fill').style.background = probElem.style.color;

  document.getElementById('detail-contract-val').textContent = `${cust.subscription_plan || 'Basic'} Plan`;
  document.getElementById('detail-tenure-val').textContent = `Tenure: ${cust.tenure_months || 0} mos • Trial Convert: ${cust.free_trial_converted || 'Yes'}`;
  document.getElementById('detail-monthly-val').textContent = `$${Number(cust.monthly_price || 8.99).toFixed(2)}`;
  document.getElementById('detail-total-val').textContent = `Downloads: ${cust.downloads_count || 0} • Devices: ${cust.number_of_devices || 1} • Profiles: ${cust.number_of_profiles || 1}`;
  document.getElementById('detail-tickets-val').textContent = `${Number(cust.watch_hours_last_30_days || 0).toFixed(1)} hrs streamed`;
  document.getElementById('detail-last-active-val').textContent = `Last watch: ${cust.days_since_last_watch || 0}d ago • Declines: ${cust.payment_failures || 0}`;

  // Render SHAP Feature Impact Chart
  renderShapChart(score.top_risk_drivers || [], score.top_protective_factors || []);

  // Natural Language Summary
  document.getElementById('detail-explanation-summary').textContent =
    score.explanation_summary || recs.retention_strategy_summary || 'Subscriber exhibits standard engagement patterns.';

  // Suggested Interventions Playbook
  const suggsContainer = document.getElementById('detail-suggestions-container');
  const actions = recs.suggested_actions || [];

  if (actions.length === 0) {
    suggsContainer.innerHTML = '<p class="text-muted">No specific interventions recommended at this time.</p>';
  } else {
    let sHtml = '';
    actions.forEach((act, idx) => {
      const isPrimary = idx === 0;
      const urgencyClass = (act.urgency || 'medium').toLowerCase();
      const triggersHtml = (act.triggered_by || []).map(t => `<span class="trigger-pill">${t}</span>`).join('');

      sHtml += `
        <div class="suggestion-card ${isPrimary ? 'primary' : ''}">
          <div class="sugg-head">
            <span class="sugg-title">${isPrimary ? '&#9733; ' : ''}${act.title}</span>
            <span class="urgency-badge ${urgencyClass}">${act.urgency}</span>
          </div>
          <p class="sugg-desc">${act.description}</p>
          <div class="sugg-triggers">${triggersHtml}</div>
        </div>
      `;
    });
    suggsContainer.innerHTML = sHtml;
  }
}

function renderShapChart(riskDrivers, protectiveFactors) {
  const ctx = document.getElementById('chart-shap-breakdown').getContext('2d');
  if (state.charts.shap) state.charts.shap.destroy();

  const combined = [];
  riskDrivers.forEach(r => combined.push({ feature: r.feature, impact: Math.abs(r.impact || 0.1), isRisk: true }));
  protectiveFactors.forEach(p => combined.push({ feature: p.feature, impact: -Math.abs(p.impact || 0.1), isRisk: false }));

  if (combined.length === 0) {
    combined.push({ feature: 'days_since_last_watch', impact: 0.54, isRisk: true });
    combined.push({ feature: 'payment_failures', impact: 0.45, isRisk: true });
    combined.push({ feature: 'watch_hours_last_30_days', impact: -0.75, isRisk: false });
    combined.push({ feature: 'tenure_months', impact: -0.47, isRisk: false });
  }

  // Format feature names
  const labels = combined.map(c => c.feature.replace(/_/g, ' '));
  const data = combined.map(c => c.impact);
  const colors = combined.map(c => c.isRisk ? '#ef4444' : '#10b981');

  state.charts.shap = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Feature Attribution Impact',
        data: data,
        backgroundColor: colors,
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#94a3b8' }
        },
        y: {
          grid: { display: false },
          ticks: { color: '#cbd5e1' }
        }
      }
    }
  });
}

function renderDetailFallback(customerId) {
  renderCustomerDetailModal({
    customer: {
      customer_id: customerId,
      subscription_plan: 'Basic',
      monthly_price: 8.99,
      watch_hours_last_30_days: 1.5,
      days_since_last_watch: 28,
      number_of_devices: 1,
      number_of_profiles: 1,
      downloads_count: 0,
      customer_support_tickets: 3,
      payment_failures: 2,
      free_trial_converted: 'Yes',
      tenure_months: 3
    },
    latest_score: {
      churn_probability: 0.942,
      churn_probability_pct: '94.2%',
      risk_tier: 'Critical',
      top_risk_drivers: [
        { feature: 'payment_failures', impact: 0.55 },
        { feature: 'days_since_last_watch', impact: 0.48 },
        { feature: 'customer_support_tickets', impact: 0.28 }
      ],
      top_protective_factors: [
        { feature: 'free_trial_converted', impact: -0.15 }
      ],
      explanation_summary: 'CRITICAL RISK (94.2% churn probability). Key triggers: recurring billing failures, 28 days of viewing dormancy, and recent playback complaints.'
    },
    current_recommendations: {
      suggested_actions: [
        {
          action_type: 'proactive_billing_support',
          title: 'Proactive Billing Support & 1-Click Grace Recovery',
          description: 'Deploy automated smart dunning email & in-app prompt offering 7-day streaming grace while updating card.',
          urgency: 'Immediate',
          triggered_by: ['payment_failures: 2 declines']
        },
        {
          action_type: 'pause_subscription_offer',
          title: 'Pause Subscription Offer (1–3 Months Free Hold)',
          description: 'Deploy retention pause modal to preserve watchlist history instead of complete account cancellation.',
          urgency: 'Immediate',
          triggered_by: ['risk_tier: Critical']
        }
      ]
    },
    interventions: []
  });
}

// ==========================================================================
// VIEW 4: Interventions Hub
// ==========================================================================

async function loadInterventions() {
  const container = document.getElementById('interventions-feed-list');
  if (!container) return;

  try {
    const res = await fetch(`${API_BASE}/interventions`);
    if (res.ok) {
      const data = await res.json();
      state.interventions = data || [];
      renderInterventionsFeed(state.interventions);
      document.getElementById('badge-interventions-count').textContent = state.interventions.length;
    } else {
      renderInterventionsFallback();
    }
  } catch (err) {
    renderInterventionsFallback();
  }
}

function renderInterventionsFeed(items) {
  const container = document.getElementById('interventions-feed-list');
  if (!container) return;

  if (items.length === 0) {
    container.innerHTML = '<p class="text-muted">No retention interventions recorded.</p>';
    return;
  }

  let html = '';
  items.forEach(item => {
    const statusClass = (item.status || 'pending').toLowerCase();
    html += `
      <div class="intervention-card glass-subcard">
        <div class="int-header">
          <div>
            <span class="int-cust">${item.customer_id}</span>
            <h4 class="int-title">${item.title}</h4>
          </div>
          <span class="status-badge ${statusClass}">${item.status}</span>
        </div>
        <p class="int-desc">${item.description}</p>
        <div class="int-meta">
          <span>Channel: <strong>${item.recommended_channel || 'in-app'}</strong></span>
          <span>By: <strong>${item.performed_by || 'Retention AI'}</strong></span>
        </div>
      </div>
    `;
  });
  container.innerHTML = html;
}

function renderInterventionsFallback() {
  const sample = [
    { id: 1, customer_id: 'OTT-98241', title: 'Proactive Billing Support & 1-Click Grace Recovery', description: 'Dispatched automated 7-day streaming grace notice to resolve expired payment card.', recommended_channel: 'in-app & email', status: 'Resolved', performed_by: 'Billing Concierge' },
    { id: 2, customer_id: 'OTT-48962', title: 'Pause Subscription Offer (1–3 Months Free Hold)', description: 'Offered 60-day pause instead of cancellation; subscriber preserved saved profile watchlists.', recommended_channel: 'in-app modal', status: 'Responded', performed_by: 'AI Retention Engine' },
    { id: 3, customer_id: 'OTT-73910', title: 'Personalized Content Discovery & Trending Watchlist Email', description: 'Sent personalized watchlist featuring newly released Sci-Fi series matching past binge history.', recommended_channel: 'email', status: 'Sent', performed_by: 'Content Recommender' }
  ];
  renderInterventionsFeed(sample);
}

// ==========================================================================
// Event Listeners & Modals
// ==========================================================================

function initEventListeners() {
  // Plan dropdown filter
  const planSelect = document.getElementById('filter-contract-type');
  if (planSelect) {
    planSelect.addEventListener('change', (e) => {
      state.filters.contract = e.target.value;
      loadCustomers();
    });
  }

  // Risk tier filter segmented control
  const tierContainer = document.getElementById('filter-risk-tier');
  if (tierContainer) {
    tierContainer.querySelectorAll('.seg-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        tierContainer.querySelectorAll('.seg-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.filters.riskTier = btn.getAttribute('data-tier') || '';
        loadCustomers();
      });
    });
  }

  // Search input
  const searchInput = document.getElementById('customer-search-input');
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      state.filters.search = e.target.value.trim();
      applyFiltersAndSort();
    });
  }

  // Sort dropdown
  const sortSelect = document.getElementById('customer-sort-by');
  if (sortSelect) {
    sortSelect.addEventListener('change', (e) => {
      state.filters.sortBy = e.target.value;
      applyFiltersAndSort();
    });
  }

  // Pagination buttons
  const btnPrev = document.getElementById('btn-prev-page');
  const btnNext = document.getElementById('btn-next-page');
  if (btnPrev) {
    btnPrev.addEventListener('click', () => {
      if (state.page > 1) {
        state.page--;
        renderCustomerTablePage();
      }
    });
  }
  if (btnNext) {
    btnNext.addEventListener('click', () => {
      const maxPage = Math.ceil(state.filteredCustomers.length / state.pageSize);
      if (state.page < maxPage) {
        state.page++;
        renderCustomerTablePage();
      }
    });
  }

  // CSV Export
  const btnExport = document.getElementById('btn-export-customers-csv');
  if (btnExport) {
    btnExport.addEventListener('click', exportFilteredCustomersToCSV);
  }
}

function exportFilteredCustomersToCSV() {
  if (state.filteredCustomers.length === 0) return;
  let csv = 'customer_id,subscription_plan,monthly_price,watch_hours_last_30_days,days_since_last_watch,customer_support_tickets,payment_failures,churn_probability,risk_tier\n';
  state.filteredCustomers.forEach(c => {
    csv += `${c.customer_id},"${c.subscription_plan}",${c.monthly_price},${c.watch_hours_last_30_days},${c.days_since_last_watch},${c.customer_support_tickets},${c.payment_failures},${c.churn_probability},${c.risk_tier}\n`;
  });
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `ott_subscribers_${Date.now()}.csv`;
  a.click();
}

function openModal(modalId) {
  const m = document.getElementById(modalId);
  if (m) m.classList.add('active');
}

function closeModal(modalId) {
  const m = document.getElementById(modalId);
  if (m) m.classList.remove('active');
}

// ==========================================================================
// REAL-TIME EVENT STREAM SIMULATOR & POLLING LAYER
// (Simulated pipeline for demonstration of real-time score updates)
// ==========================================================================

let simPollingInterval = null;
let isSimulatorRunning = false;

function initLiveSimulation() {
  checkSimulatorStatus();
  if (!simPollingInterval) {
    simPollingInterval = setInterval(pollLiveUpdates, 4000);
  }
}

async function checkSimulatorStatus() {
  try {
    const res = await fetch(`${API_BASE}/simulate/live-events/status`);
    if (res.ok) {
      const data = await res.json();
      updateSimulatorUI(data.is_running);
      if (data.last_event) {
        displaySimulatedEventInBanner(data.last_event);
      }
    }
  } catch (e) {
    console.warn('Could not reach simulator status:', e);
  }
}

async function toggleLiveSimulation() {
  const endpoint = isSimulatorRunning ? '/simulate/live-events/stop' : '/simulate/live-events/start';
  try {
    const res = await fetch(`${API_BASE}${endpoint}`, { method: 'POST' });
    if (res.ok) {
      const data = await res.json();
      updateSimulatorUI(data.is_running);
      if (data.is_running) {
        setBannerText('⚡ Live streaming event generator STARTED. Synthetic events will be ingested every 3.5s.', 'live-event');
      } else {
        setBannerText('Live streaming event generator PAUSED.', 'normal');
      }
    }
  } catch (err) {
    alert('Error toggling live simulator: ' + err.message);
  }
}

function updateSimulatorUI(running) {
  isSimulatorRunning = running;
  const btn = document.getElementById('btn-toggle-sim');
  const btnText = document.getElementById('btn-toggle-sim-text');
  if (btn && btnText) {
    if (running) {
      btn.classList.add('running');
      btnText.textContent = 'Stop Stream';
    } else {
      btn.classList.remove('running');
      btnText.textContent = 'Start Live Stream';
    }
  }
}

async function stepLiveEvent() {
  const stepBtn = document.getElementById('btn-step-sim');
  if (stepBtn) stepBtn.disabled = true;
  try {
    const res = await fetch(`${API_BASE}/simulate/live-events/step`, { method: 'POST' });
    if (res.ok) {
      const data = await res.json();
      const prevProb = data.previous_score?.churn_probability || 0;
      const newProb = data.new_score?.churn_probability || 0;
      const prevTier = data.previous_score?.risk_tier || 'Unknown';
      const newTier = data.new_score?.risk_tier || 'Unknown';

      const evtSummary = {
        customer_id: data.customer_id,
        event_type: data.event_type,
        timestamp: new Date().toLocaleTimeString(),
        prev_prob: prevProb,
        new_prob: newProb,
        prev_tier: prevTier,
        new_tier: newTier,
        escalated: data.escalated_to_critical_queue,
        message: data.message
      };
      displaySimulatedEventInBanner(evtSummary);

      // Immediately refresh active view
      await pollLiveUpdates();
    }
  } catch (err) {
    console.error('Error triggering sample event:', err);
  } finally {
    if (stepBtn) stepBtn.disabled = false;
  }
}

function displaySimulatedEventInBanner(evt) {
  const banner = document.getElementById('live-event-banner');
  const msgEl = document.getElementById('live-banner-message');
  const timeEl = document.getElementById('live-banner-time');
  if (!banner || !msgEl) return;

  const prevPct = (evt.prev_prob * 100).toFixed(1);
  const newPct = (evt.new_prob * 100).toFixed(1);
  const delta = (evt.new_prob - evt.prev_prob) * 100;
  const deltaBadge = delta >= 0 ? `+${delta.toFixed(1)}%` : `${delta.toFixed(1)}%`;
  const deltaColor = delta >= 0 ? '#ef4444' : '#10b981';

  let text = `<strong>[${evt.event_type}]</strong> for <span style="color:#60a5fa">${evt.customer_id}</span>: Score ${prevPct}% (${evt.prev_tier}) &rarr; <strong>${newPct}% (${evt.new_tier})</strong> <span style="color:${deltaColor}">[${deltaBadge}]</span>`;
  if (evt.escalated) {
    text += ` &bull; <span style="color:#ef4444; font-weight:bold;">🚨 ESCALATED TO CRITICAL QUEUE!</span>`;
    banner.classList.add('critical-event');
  } else {
    banner.classList.remove('critical-event');
    banner.classList.add('active-event');
  }

  msgEl.innerHTML = text;
  if (timeEl) timeEl.textContent = evt.timestamp || new Date().toLocaleTimeString();

  setTimeout(() => {
    banner.classList.remove('active-event');
  }, 2500);
}

function setBannerText(text, stateClass) {
  const banner = document.getElementById('live-event-banner');
  const msgEl = document.getElementById('live-banner-message');
  if (msgEl) msgEl.innerHTML = text;
  if (banner) {
    banner.className = 'live-event-banner';
    if (stateClass) banner.classList.add(stateClass);
  }
}

async function pollLiveUpdates() {
  try {
    // 1. Check simulator status
    const simRes = await fetch(`${API_BASE}/simulate/live-events/status`);
    if (simRes.ok) {
      const simData = await simRes.json();
      updateSimulatorUI(simData.is_running);
      if (simData.last_event) {
        displaySimulatedEventInBanner(simData.last_event);
      }
    }

    // 2. Poll overview analytics (silent refresh)
    if (state.activePage === 'overview') {
      const ovRes = await fetch(`${API_BASE}/analytics/overview`);
      if (ovRes.ok) {
        const ovData = await ovRes.json();
        renderOverviewKPIs(ovData);
        renderRiskDonutChart(ovData.risk_distribution);
        renderSegmentBars(ovData.subscription_plan_distribution || ovData.contract_distribution);
      }
    }

    // 3. Poll customer directory and priority queue silently
    const params = new URLSearchParams();
    if (state.filters.riskTier) params.append('risk_tier', state.filters.riskTier);
    if (state.filters.contract) params.append('subscription_plan', state.filters.contract);
    if (state.filters.search) params.append('search', state.filters.search);
    params.append('limit', '250');

    const custRes = await fetch(`${API_BASE}/customers?${params.toString()}`);
    if (custRes.ok) {
      const custData = await custRes.json();
      state.customers = custData.customers || [];
      state.totalCustomers = custData.total_customers || state.customers.length;

      // Update critical attention queue on overview
      renderPriorityFeed(state.customers);

      // Re-apply filters for table
      let list = [...state.customers];
      if (state.filters.riskTier) {
        list = list.filter(c => c.risk_tier && c.risk_tier.toLowerCase() === state.filters.riskTier.toLowerCase());
      }
      if (state.filters.contract) {
        list = list.filter(c => c.subscription_plan === state.filters.contract);
      }
      if (state.filters.search) {
        const q = state.filters.search.toLowerCase();
        list = list.filter(c => c.customer_id.toLowerCase().includes(q));
      }
      list.sort((a, b) => {
        switch (state.filters.sortBy) {
          case 'risk-desc': return (b.churn_probability || 0) - (a.churn_probability || 0);
          case 'risk-asc': return (a.churn_probability || 0) - (b.churn_probability || 0);
          case 'price-desc': return (b.monthly_price || 0) - (a.monthly_price || 0);
          case 'watch-desc': return (b.watch_hours_last_30_days || 0) - (a.watch_hours_last_30_days || 0);
          default: return 0;
        }
      });
      state.filteredCustomers = list;
      if (state.activePage === 'customers') {
        renderCustomerTablePage();
      }
    }
  } catch (err) {
    // Silent fail in polling
  }
}
