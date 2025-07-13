'use strict';
document.addEventListener('DOMContentLoaded', function () {
  setTimeout(function () {
    var options = {
      chart: { type: 'bar', height: 100, sparkline: { enabled: true } },
      colors: ['#1677ff'],
      plotOptions: { bar: { columnWidth: '80%' } },
      series: [
        {
          data: [
            220, 230, 240, 220, 225, 215, 205, 195, 185, 150, 185, 195, 80, 205, 215, 225, 240, 225, 215, 205, 80, 215, 225, 240, 215, 210,
            190
          ]
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
    var chart = new ApexCharts(document.querySelector('#total-value-graph-1'), options);
    chart.render();
  }, 500);
});
