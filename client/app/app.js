var version = "v0.0.1";

$(document).ready( function () {
    $('#table').DataTable( {
        ajax: {
            // url: '/github_data_local.json',
            url: 'http://prd-s3-crazy-awesome-python-api.s3-website-us-east-1.amazonaws.com/github_data.json',
            dataSrc: 'data'
        },
        columns: [
          { data: "category", title: "category" },
          { data: "_description", title: "description" },
          { data: "githuburl", title: "url" },
          { data: "_stars", title: "stars" },
          { data: "_stars_per_week", title: "stars/wk" },
          { data: "_forks", title: "forks" },
        ],
        paging: false,
    });
});



