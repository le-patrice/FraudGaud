'use strict';
document.addEventListener('DOMContentLoaded', function () {
  setTimeout(function () {
    var options = {
      chart: { type: 'area', height: 100, sparkline: { enabled: true } },
      colors: ['#1677ff'],
      plotOptions: { bar: { columnWidth: '80%' } },
      series: [
        {
          data: [100, 140, 100, 240, 115, 125, 90, 100, 80, 150, 160, 150, 170, 155, 150, 160, 145, 200, 140, 160]
        }
      ],
      xaxis: { crosshairs: { width: 1 } },
      stroke: {
        curve: 'straight',
        width: 1.5
      },
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
    var chart = new ApexCharts(document.querySelector('#total-value-graph-4'), options);
    chart.render();
  }, 500);
});
