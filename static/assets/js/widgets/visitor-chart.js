'use strict';
document.addEventListener('DOMContentLoaded', function () {
  setTimeout(function () {
    var options = {
      chart: {
        height: 450,
        type: 'area',
        toolbar: {
          show: false
        }
      },
      dataLabels: {
        enabled: false
      },
      colors: ['#1677ff', '#13c2c2'],
      series: [
        {
          name: 'Page Views',
          data: [31, 40, 28, 51, 42, 109, 100]
        },
        {
          name: 'Sessions',
          data: [11, 32, 45, 32, 34, 52, 41]
        }
      ],
      stroke: {
        curve: 'smooth',
        width: 2
      },
      xaxis: {
        categories: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
      }
    };
    var chart = new ApexCharts(document.querySelector('#visitor-chart'), options);
    chart.render();
    var options1 = {
      chart: {
        height: 450,
        type: 'area',
        toolbar: {
          show: false
        }
      },
      dataLabels: {
        enabled: false
      },
      colors: ['#1677ff', '#13c2c2'],
      series: [
        {
          name: 'Page Views',
          data: [76, 85, 101, 98, 87, 105, 91, 114, 94, 86, 115, 35]
        },
        {
          name: 'Sessions',
          data: [110, 60, 150, 35, 60, 36, 26, 45, 65, 52, 53, 41]
        }
      ],
      stroke: {
        curve: 'smooth',
        width: 2
      },
      xaxis: {
        categories: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
      }
    };
    var chart = new ApexCharts(document.querySelector('#visitor-chart-1'), options1);
    chart.render();
  }, 500);
});
