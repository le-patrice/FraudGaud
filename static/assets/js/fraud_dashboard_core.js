/*!
 * FraudGuard Analytics Dashboard - Core JavaScript
 * Advanced ML-Powered Fraud Detection System
 * Version: 1.0.0
 * Author: FraudGuard Team
 */

// =============================================================================
// DASHBOARD CORE CONFIGURATION
// =============================================================================

const FraudGuardConfig = {
    version: '1.0.0',
    apiBaseUrl: '/api/v1/',
    refreshInterval: 30000, // 30 seconds
    chartAnimationDuration: 1000,
    realTimeEnabled: true,
    debugMode: false,
    
    // Model Performance Data - Real Results
    modelPerformance: {
        'Logistic Regression': {
            accuracy: 99.92,
            precision: 83.12,
            recall: 65.31,
            f1: 73.14,
            rocAuc: 95.60
        },
        'Random Forest': {
            accuracy: 99.96,
            precision: 94.12,
            recall: 81.63,
            f1: 87.43,
            rocAuc: 96.30
        },
        'XGBoost': {
            accuracy: 99.71,
            precision: 36.13,
            recall: 87.76,
            f1: 51.19,
            rocAuc: 97.65
        },
        'SVM': {
            accuracy: 99.89,
            precision: 94.45,
            recall: 78.57,
            f1: 85.96,
            rocAuc: 97.34
        }
    },
    
    // Feature Importance Data
    featureImportance: {
        'V17': 17.03,
        'V14': 13.64,
        'V12': 13.33,
        'V10': 7.41,
        'V16': 7.18,
        'V11': 4.52,
        'V9': 3.14,
        'V4': 3.02,
        'V18': 2.81,
        'V7': 2.53
    },
    
    // Uganda Market Data
    ugandaData: {
        exchangeRate: 3700,
        mobileMoneyUsers: 25000000,
        bankAccountHolders: 12000000,
        fraudLossEstimate: 50000000,
        majorProviders: ['MTN Mobile Money', 'Airtel Money', 'Stanbic Bank', 'Centenary Bank']
    }
};

// =============================================================================
// DASHBOARD STATE MANAGEMENT
// =============================================================================

class FraudGuardState {
    constructor() {
        this.data = {
            liveTransactions: [],
            fraudAlerts: [],
            businessMetrics: {},
            systemStatus: 'online',
            lastUpdate: new Date(),
            activeSessions: 0,
            totalProcessed: 0
        };
        
        this.charts = {};
        this.intervals = {};
        this.isRealTimeEnabled = true;
    }
    
    updateMetric(key, value) {
        this.data.businessMetrics[key] = value;
        this.triggerUpdate('metrics', { key, value });
    }
    
    addTransaction(transaction) {
        this.data.liveTransactions.unshift(transaction);
        if (this.data.liveTransactions.length > 100) {
            this.data.liveTransactions = this.data.liveTransactions.slice(0, 100);
        }
        this.triggerUpdate('transactions', transaction);
    }
    
    addAlert(alert) {
        this.data.fraudAlerts.unshift(alert);
        this.triggerUpdate('alerts', alert);
        this.updateAlertBadges();
    }
    
    triggerUpdate(type, data) {
        document.dispatchEvent(new CustomEvent('fraudguard:update', {
            detail: { type, data }
        }));
    }
    
    updateAlertBadges() {
        const alertCount = this.data.fraudAlerts.filter(alert => !alert.read).length;
        document.querySelectorAll('#alert-count, #header-alert-count').forEach(badge => {
            if (badge) {
                badge.textContent = alertCount;
                badge.classList.toggle('fraud-alert-badge', alertCount > 0);
            }
        });
    }
}

// Initialize global state
const fraudGuardState = new FraudGuardState();

// =============================================================================
// CHART GENERATION & MANAGEMENT
// =============================================================================

class ChartManager {
    constructor() {
        this.charts = {};
        this.defaultOptions = {
            chart: {
                fontFamily: 'Public Sans, sans-serif',
                toolbar: { show: false },
                animations: {
                    enabled: true,
                    easing: 'easeinout',
                    speed: 800
                }
            },
            colors: ['#5e72e4', '#2dce89', '#fb6340', '#11cdef', '#f5365c'],
            theme: { mode: 'light' }
        };
    }
    
    createModelPerformanceChart() {
        const models = Object.keys(FraudGuardConfig.modelPerformance);
        const accuracyData = models.map(model => 
            FraudGuardConfig.modelPerformance[model].accuracy
        );
        
        const options = {
            ...this.defaultOptions,
            series: [{
                name: 'Accuracy',
                data: accuracyData
            }],
            chart: {
                ...this.defaultOptions.chart,
                type: 'bar',
                height: 350
            },
            xaxis: {
                categories: models,
                labels: {
                    style: { colors: '#8392ab', fontSize: '12px' }
                }
            },
            yaxis: {
                min: 99,
                max: 100,
                labels: {
                    formatter: (val) => `${val.toFixed(2)}%`,
                    style: { colors: '#8392ab' }
                }
            },
            dataLabels: {
                enabled: true,
                formatter: (val) => `${val.toFixed(2)}%`,
                style: { colors: ['#fff'], fontSize: '12px', fontWeight: 'bold' }
            },
            plotOptions: {
                bar: {
                    borderRadius: 8,
                    dataLabels: { position: 'top' }
                }
            },
            title: {
                text: 'Model Accuracy Comparison',
                style: { color: '#32325d', fontSize: '16px', fontWeight: '600' }
            }
        };
        
        this.renderChart('model-performance-chart', options);
    }
    
    createComprehensiveMetricsChart() {
        const models = Object.keys(FraudGuardConfig.modelPerformance);
        const metrics = ['accuracy', 'precision', 'recall', 'f1'];
        
        const series = metrics.map(metric => ({
            name: metric.charAt(0).toUpperCase() + metric.slice(1),
            data: models.map(model => FraudGuardConfig.modelPerformance[model][metric])
        }));
        
        const options = {
            ...this.defaultOptions,
            series,
            chart: {
                ...this.defaultOptions.chart,
                type: 'bar',
                height: 350
            },
            xaxis: {
                categories: models,
                labels: {
                    style: { colors: '#8392ab', fontSize: '11px' }
                }
            },
            yaxis: {
                labels: {
                    formatter: (val) => `${val.toFixed(1)}%`,
                    style: { colors: '#8392ab' }
                }
            },
            dataLabels: { enabled: false },
            legend: {
                position: 'top',
                horizontalAlign: 'left',
                labels: { colors: '#8392ab' }
            },
            plotOptions: {
                bar: {
                    borderRadius: 4,
                    columnWidth: '60%'
                }
            }
        };
        
        this.renderChart('model-metrics-chart', options);
    }
    
    createKPISparklines() {
        // Accuracy trend
        this.createSparkline('fraud-accuracy-chart', [99.91, 99.93, 99.94, 99.95, 99.96, 99.96, 99.96], '#5e72e4');
        
        // Detection trend
        this.createSparkline('fraud-detection-chart', [79.2, 80.1, 80.8, 81.1, 81.4, 81.6, 81.63], '#2dce89');
        
        // False positive trend
        this.createSparkline('false-positive-chart', [0.012, 0.011, 0.010, 0.010, 0.009, 0.009, 0.009], '#fb6340', 'bar');
        
        // Savings trend
        this.createSparkline('savings-chart', [8420, 8890, 9120, 9340, 9480, 9580, 9651], '#11cdef');
    }
    
    createSparkline(elementId, data, color, type = 'line') {
        const options = {
            series: [{
                data: data
            }],
            chart: {
                type: type,
                height: 60,
                sparkline: { enabled: true },
                animations: { enabled: true, speed: 400 }
            },
            colors: [color],
            stroke: {
                width: type === 'line' ? 2 : 0,
                curve: 'smooth'
            },
            fill: {
                type: type === 'line' ? 'gradient' : 'solid',
                gradient: {
                    shadeIntensity: 1,
                    opacityFrom: 0.7,
                    opacityTo: 0.1
                }
            },
            tooltip: { enabled: false }
        };
        
        this.renderChart(elementId, options);
    }
    
    createBusinessImpactChart() {
        const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
        const savings = [8420, 8890, 9120, 9340, 9480, 9580, 9651];
        
        const options = {
            ...this.defaultOptions,
            series: [{
                name: 'Daily Savings',
                data: savings
            }],
            chart: {
                ...this.defaultOptions.chart,
                type: 'area',
                height: 200
            },
            xaxis: {
                categories: days,
                labels: { show: false }
            },
            yaxis: { labels: { show: false } },
            dataLabels: { enabled: false },
            stroke: { width: 2, curve: 'smooth' },
            fill: {
                type: 'gradient',
                gradient: {
                    shadeIntensity: 1,
                    opacityFrom: 0.7,
                    opacityTo: 0.1
                }
            },
            grid: { show: false },
            tooltip: {
                x: { format: 'dddd' },
                y: { formatter: (val) => `$${val.toLocaleString()}` }
            }
        };
        
        this.renderChart('business-impact-chart', options);
    }
    
    createUgandaSavingsChart() {
        const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
        const ugxSavings = [28500000, 31200000, 33800000, 34500000, 35100000, 35400000, 35750000];
        
        const options = {
            ...this.defaultOptions,
            series: [{
                name: 'UGX Savings',
                data: ugxSavings
            }],
            chart: {
                ...this.defaultOptions.chart,
                type: 'line',
                height: 200
            },
            colors: ['#FF6B35'],
            xaxis: {
                categories: days,
                labels: {
                    style: { colors: '#8392ab', fontSize: '10px' }
                }
            },
            yaxis: {
                labels: {
                    formatter: (val) => `${(val / 1000000).toFixed(1)}M`,
                    style: { colors: '#8392ab', fontSize: '10px' }
                }
            },
            stroke: { width: 3, curve: 'smooth' },
            markers: { size: 6, colors: ['#FF6B35'], strokeWidth: 2, strokeColors: '#fff' },
            tooltip: {
                y: { formatter: (val) => `UGX ${val.toLocaleString()}` }
            },
            grid: {
                borderColor: 'rgba(224, 230, 237, 0.3)',
                strokeDashArray: 3
            }
        };
        
        this.renderChart('uganda-savings-chart', options);
    }
    
    renderChart(elementId, options) {
        const element = document.getElementById(elementId);
        if (!element) {
            console.warn(`Chart element ${elementId} not found`);
            return;
        }
        
        // Check if ApexCharts is available
        if (typeof ApexCharts === 'undefined') {
            console.warn('ApexCharts not loaded, showing placeholder');
            element.innerHTML = '<div class="text-center text-muted p-4">Chart requires ApexCharts library</div>';
            return;
        }
        
        // Remove loading state
        element.innerHTML = '';
        element.classList.remove('chart-loading');
        
        try {
            const chart = new ApexCharts(element, options);
            chart.render();
            this.charts[elementId] = chart;
        } catch (error) {
            console.error(`Error rendering chart ${elementId}:`, error);
            element.innerHTML = `<div class="text-center text-muted p-4">Chart unavailable</div>`;
        }
    }
    
    refreshAllCharts() {
        Object.values(this.charts).forEach(chart => {
            if (chart && chart.updateOptions) {
                chart.updateOptions({
                    chart: { animations: { enabled: true } }
                });
            }
        });
    }
}

// Initialize chart manager
const chartManager = new ChartManager();

// =============================================================================
// REAL-TIME DATA SIMULATION
// =============================================================================

class RealTimeDataManager {
    constructor() {
        this.isActive = true;
        this.transactionInterval = null;
        this.alertInterval = null;
        this.metricsInterval = null;
    }
    
    start() {
        this.isActive = true;
        
        // Generate transactions every 5 seconds
        this.transactionInterval = setInterval(() => {
            this.generateTransaction();
        }, 5000);
        
        // Generate alerts every 30 seconds
        this.alertInterval = setInterval(() => {
            if (Math.random() < 0.3) { // 30% chance
                this.generateAlert();
            }
        }, 30000);
        
        // Update metrics every 10 seconds
        this.metricsInterval = setInterval(() => {
            this.updateMetrics();
        }, 10000);
        
        console.log('🔄 Real-time data manager started');
    }
    
    stop() {
        this.isActive = false;
        [this.transactionInterval, this.alertInterval, this.metricsInterval].forEach(interval => {
            if (interval) clearInterval(interval);
        });
        console.log('⏸️ Real-time data manager stopped');
    }
    
    generateTransaction() {
        const merchants = [
            'Amazon.com', 'Starbucks', 'Shell Uganda', 'MTN Mobile Money',
            'Airtel Money', 'Best Buy', 'Shoprite', 'Garden City Mall',
            'Unknown Merchant', 'Crypto Exchange', 'Foreign Website'
        ];
        
        const locations = ['Kampala', 'Entebbe', 'Jinja', 'Mbarara', 'Unknown', 'Foreign'];
        
        const isFraud = Math.random() < 0.01; // 1% fraud rate
        const amount = isFraud ? 
            Math.random() * 2000 + 500 : // $500-$2500 for fraud
            Math.random() * 500 + 10;    // $10-$510 for legitimate
        
        const merchant = merchants[Math.floor(Math.random() * (isFraud ? merchants.length : merchants.length - 3))];
        const riskScore = isFraud ? Math.random() * 0.4 + 0.6 : Math.random() * 0.3;
        
        const transaction = {
            id: Date.now(),
            time: new Date().toLocaleTimeString(),
            amount: amount,
            merchant: merchant,
            location: locations[Math.floor(Math.random() * locations.length)],
            riskScore: riskScore,
            status: riskScore > 0.7 ? 'BLOCKED' : (riskScore > 0.4 ? 'REVIEW' : 'APPROVED'),
            isFraud: isFraud,
            timestamp: new Date()
        };
        
        fraudGuardState.addTransaction(transaction);
        this.updateTransactionTable(transaction);
        
        // Update live count
        const liveCountElement = document.getElementById('live-count');
        if (liveCountElement) {
            const current = parseInt(liveCountElement.textContent) || 0;
            liveCountElement.textContent = current + 1;
        }
    }
    
    generateAlert() {
        const alertTypes = [
            {
                type: 'high_risk',
                title: 'High Risk Transaction Detected',
                description: `$${(Math.random() * 2000 + 500).toFixed(0)} - Unknown merchant`,
                icon: 'ph-warning-circle',
                class: 'bg-light-danger text-danger',
                action: 'BLOCKED'
            },
            {
                type: 'unusual_pattern',
                title: 'Unusual Transaction Pattern',
                description: 'Multiple locations in short timeframe',
                icon: 'ph-clock',
                class: 'bg-light-warning text-warning',
                action: 'REVIEW'
            },
            {
                type: 'model_update',
                title: 'Model Performance Update',
                description: 'Accuracy improved to 99.96%',
                icon: 'ph-shield-check',
                class: 'bg-light-success text-success',
                action: 'SUCCESS'
            }
        ];
        
        const alertType = alertTypes[Math.floor(Math.random() * alertTypes.length)];
        const alert = {
            ...alertType,
            id: Date.now(),
            timestamp: new Date(),
            timeAgo: 'just now',
            read: false
        };
        
        fraudGuardState.addAlert(alert);
        this.updateAlertsDropdown();
    }
    
    updateTransactionTable(transaction) {
        const tbody = document.getElementById('live-transactions-tbody');
        if (!tbody) return;
        
        const row = document.createElement('tr');
        row.className = 'transaction-row-new';
        
        const statusClass = transaction.status === 'BLOCKED' ? 'text-danger' :
                           transaction.status === 'REVIEW' ? 'text-warning' : 'text-success';
        
        const statusIcon = transaction.status === 'BLOCKED' ? '✗' :
                          transaction.status === 'REVIEW' ? '⚠' : '✓';
        
        row.innerHTML = `
            <td>${transaction.time}</td>
            <td>$${transaction.amount.toFixed(2)}</td>
            <td>${transaction.merchant}</td>
            <td><span class="fraud-score-${transaction.riskScore > 0.7 ? 'high' : transaction.riskScore > 0.4 ? 'medium' : 'low'}">${transaction.riskScore.toFixed(3)}</span></td>
            <td><span class="${statusClass}">${statusIcon} ${transaction.status}</span></td>
            <td class="text-end">$${transaction.status === 'BLOCKED' ? '0.00' : transaction.amount.toFixed(2)}</td>
        `;
        
        tbody.insertBefore(row, tbody.firstChild);
        
        // Remove old rows (keep only 10)
        while (tbody.children.length > 10) {
            tbody.removeChild(tbody.lastChild);
        }
        
        // Remove animation class after animation completes
        setTimeout(() => {
            row.classList.remove('transaction-row-new');
        }, 1000);
    }
    
    updateAlertsDropdown() {
        const container = document.getElementById('alerts-container');
        if (!container) return;
        
        const alerts = fraudGuardState.data.fraudAlerts.slice(0, 5);
        
        container.innerHTML = alerts.map(alert => `
            <a class="list-group-item list-group-item-action ${alert.read ? 'opacity-50' : ''}" href="#" onclick="markAlertRead('${alert.id}')">
                <div class="d-flex">
                    <div class="flex-shrink-0">
                        <div class="user-avtar ${alert.class}">
                            <i class="${alert.icon}"></i>
                        </div>
                    </div>
                    <div class="flex-grow-1 ms-1">
                        <span class="float-end text-muted text-sm">${alert.timeAgo}</span>
                        <p class="text-body mb-1">${alert.title}</p>
                        <span class="text-muted text-sm">${alert.description}</span>
                    </div>
                </div>
            </a>
        `).join('');
    }
    
    updateMetrics() {
        // Update last update time
        const lastUpdateElement = document.getElementById('last-update');
        if (lastUpdateElement) {
            lastUpdateElement.textContent = 'just now';
        }
        
        // Simulate small variations in metrics
        const variations = {
            accuracy: 99.96 + (Math.random() - 0.5) * 0.01,
            detection: 81.63 + (Math.random() - 0.5) * 0.5,
            falsePositive: 0.009 + (Math.random() - 0.5) * 0.001,
            savings: 9651 + Math.floor((Math.random() - 0.5) * 100)
        };
        
        fraudGuardState.updateMetric('variations', variations);
    }
}

// Initialize real-time data manager
const realTimeManager = new RealTimeDataManager();

// =============================================================================
// INTERACTIVE FEATURES
// =============================================================================

function scrollToSection(sectionId) {
    const element = document.getElementById(sectionId);
    if (element) {
        element.scrollIntoView({ 
            behavior: 'smooth',
            block: 'start'
        });
        
        // Update active sidebar link
        document.querySelectorAll('.pc-link').forEach(link => {
            link.classList.remove('active');
        });
        
        const activeLink = document.querySelector(`[href="#${sectionId}"]`);
        if (activeLink) {
            activeLink.classList.add('active');
        }
    }
}

function refreshCharts() {
    // Show loading indicators
    document.querySelectorAll('[id$="-chart"]').forEach(element => {
        if (!element.classList.contains('chart-loading')) {
            element.innerHTML = '<div class="chart-loading"><div class="loading-spinner"></div></div>';
        }
    });
    
    // Refresh all charts after a delay
    setTimeout(() => {
        chartManager.createModelPerformanceChart();
        chartManager.createComprehensiveMetricsChart();
        chartManager.createKPISparklines();
        chartManager.createBusinessImpactChart();
        chartManager.createUgandaSavingsChart();
    }, 1000);
}

function markAlertRead(alertId) {
    const alert = fraudGuardState.data.fraudAlerts.find(a => a.id === alertId);
    if (alert) {
        alert.read = true;
        fraudGuardState.updateAlertBadges();
        realTimeManager.updateAlertsDropdown();
    }
}

// =============================================================================
// DASHBOARD INITIALIZATION
// =============================================================================

function initializeDashboard() {
    console.log('🚀 Initializing FraudGuard Dashboard...');
    
    // Initialize charts with delay to ensure DOM is ready
    setTimeout(() => {
        chartManager.createModelPerformanceChart();
        chartManager.createComprehensiveMetricsChart();
        chartManager.createKPISparklines();
        chartManager.createBusinessImpactChart();
        chartManager.createUgandaSavingsChart();
    }, 500);
    
    // Start real-time data simulation
    setTimeout(() => {
        realTimeManager.start();
    }, 1000);
    
    // Initialize sidebar functionality
    initializeSidebar();
    
    // Initialize search functionality
    initializeSearch();
    
    // Set up event listeners
    setupEventListeners();
    
    console.log('✅ FraudGuard Dashboard initialized successfully');
}

function initializeSidebar() {
    // Enhanced sidebar interactions
    document.querySelectorAll('.pc-link').forEach(link => {
        link.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            if (href && href.startsWith('#')) {
                e.preventDefault();
                const sectionId = href.substring(1);
                scrollToSection(sectionId);
            }
        });
    });
    
    // Submenu toggles
    document.querySelectorAll('.pc-hasmenu > .pc-link').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const parent = this.parentElement;
            const submenu = parent.querySelector('.pc-submenu');
            
            if (submenu) {
                const isActive = parent.classList.contains('active');
                
                // Close other submenus
                document.querySelectorAll('.pc-hasmenu.active').forEach(item => {
                    if (item !== parent) {
                        item.classList.remove('active');
                        const otherSubmenu = item.querySelector('.pc-submenu');
                        if (otherSubmenu) otherSubmenu.style.display = 'none';
                    }
                });
                
                // Toggle current submenu
                parent.classList.toggle('active');
                submenu.style.display = isActive ? 'none' : 'block';
            }
        });
    });
}

function initializeSearch() {
    const searchInput = document.getElementById('global-search');
    if (searchInput) {
        searchInput.addEventListener('input', function(e) {
            const query = e.target.value.toLowerCase();
            if (query.length > 2) {
                console.log('Searching for:', query);
            }
        });
    }
}

function setupEventListeners() {
    // Listen for state updates
    document.addEventListener('fraudguard:update', function(e) {
        const { type, data } = e.detail;
        console.log(`State update: ${type}`, data);
    });
    
    // Window resize handler
    window.addEventListener('resize', function() {
        setTimeout(() => {
            chartManager.refreshAllCharts();
        }, 300);
    });
    
    // Visibility change handler
    document.addEventListener('visibilitychange', function() {
        if (document.hidden) {
            realTimeManager.stop();
        } else {
            realTimeManager.start();
        }
    });
}

// =============================================================================
// GLOBAL ERROR HANDLING
// =============================================================================

window.addEventListener('error', function(e) {
    console.error('FraudGuard Dashboard Error:', e.error);
});

window.addEventListener('unhandledrejection', function(e) {
    console.error('FraudGuard Dashboard Promise Rejection:', e.reason);
});

// =============================================================================
// EXPORT FOR EXTERNAL USE
// =============================================================================

window.FraudGuard = {
    state: fraudGuardState,
    charts: chartManager,
    realTime: realTimeManager,
    config: FraudGuardConfig,
    init: initializeDashboard,
    version: FraudGuardConfig.version
};

console.log('🛡️ FraudGuard Analytics Dashboard Core Loaded');