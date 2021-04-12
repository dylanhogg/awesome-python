var version = "v0.0.1";

$(document).ready( function () {
    $('#table').DataTable( {
        ajax: {
            // url: '/github_data_local.json',
            url: 'http://prd-s3-crazy-awesome-python-api.s3-website-us-east-1.amazonaws.com/github_data.json',
            dataSrc: 'data'
        },
        order: [[ 4, "desc" ]],
        columns: [
          { data: "category", title: "Category" },
          { data: "_description", title: "Description" },
          { data: "_repopath", title: "Github",
            render: function(data, type, row, meta) { return "<a href='https://github.com/" + data + "'>" + data + "</a>"; }
          },
          { data: "_stars", title: "Stars", render: $.fn.dataTable.render.number(',', '.', 0) },
          { data: "_stars_per_week", title: "Stars\nper&nbsp;week",
            render: function(data, type, row, meta) { return data > 10 ? data.toFixed(0) : data.toFixed(1); }
          },
          { data: "_forks", title: "Forks", render: $.fn.dataTable.render.number(',', '.', 0) },
          { data: "_created_at", title: "Created",
            render: function(data, type, row, meta) { return new Date(data).toISOString().split('T')[0]; }
          },
          { data: "_age_weeks", title: "Age in&nbsp;weeks",
            render: function(data, type, row, meta) { return data.toFixed(0); }
          },
        ],
        paging: false,
    });
});



