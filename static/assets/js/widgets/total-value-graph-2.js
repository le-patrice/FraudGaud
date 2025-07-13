'use strict';
document.addEventListener('DOMContentLoaded', function () {
  setTimeout(function () {
    var options = {
      chart: { type: 'area', height: 100, sparkline: { enabled: true } },
      colors: ['#ff4d4f'],
      plotOptions: { bar: { columnWidth: '80%' } },
      series: [
        {
          data: [1800, 1500, 1800, 1700, 1400, 1200, 1000, 800, 600, 500, 600, 800, 500, 700, 400, 600, 500, 600]
        }
      ],
      xaxis: { crosshairs: { width: 1 } },
      tooltip: {
        fixed: { enabled: false },
        x: { show: false },
        y: {
          title: {
            formatter: function (seriesName) {
              return '';
            }
          }
        },
        marker: { show: false }
      }
    };
    var chart = new ApexCharts(document.querySelector('#total-value-graph-2'), options);
    chart.render();
  }, 500);
});
