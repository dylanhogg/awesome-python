const version = "v0.0.11";
const CATEGORY_COL = 0;  // 0-based
const TAG_COL = 8;

// TODO: load CATEGORY_DATA from url and sinc with app.js...
var CATEGORY_DATA = {
    '': 'Select category...',
    'all': 'All categories',
    'chatgpt': 'ChatGPT and LLMs',
    'crypto': 'Crypto',
    'data': 'Data',
    'debug': 'Debugging',
    'diffusion': 'Diffusion Text to Image',
    'finance': 'Finance',
    'gamedev': 'Game Development',
    'gis': 'GIS',
    'graph': 'Graph',
    'gui': 'GUI',
    'jupyter': 'Jupyter',
    'math': 'Math',
    'ml': 'ML - General',
    'ml-dl': 'ML - Deep Learning',
    'ml-interpretability': 'ML - Interpretability',
    'ml-ops': 'ML - Ops',
    'time-series': 'ML - Time Series',
    'nlp': 'NLP',
    'pandas': 'Pandas',
    'perf': 'Performance',
    'profiling': 'Profiling',
    'security': 'Security',
    'sim': 'Simulation',
    'study': 'Study',
    'template': 'Template',
    'term': 'Terminal',
    'testing': 'Testing',
    'typing': 'Typing',
    'util': 'Utility',
    'viz': 'Vizualisation',
    'web': 'Web',
}

$(document).ready( function () {
    $("#menu-icon").click(function(){
        // https://www.w3schools.com/howto/howto_js_sidenav.asp
        $('#sidenav').width("200px");
    });

    $("#menu-close-btn").click(function(){
        $('#sidenav').width("0px");
    });

    $("#container").click(function(){
        $('#sidenav').width("0px");
    });

    var ajax_url = (window.location.href.indexOf("hot") !== -1) ? './github_hot.json?v1.0' : './github_top.json?v1.0';
    $.get(ajax_url, function (data) {
        var tbl = $("<table />").attr("id", "top_repos_table");
        $("#top_repos").append(tbl);

        var MAX_ASSETS = 3;
        var html = "";
        var repos = data["data"];
        for(var i=0; i<repos.length; i++)
        {
            var row = repos[i];
            var rank = row.rank;
            if (rank > MAX_ASSETS) {
                continue;
            }

            var org_repo = row._organization + "/" + row._reponame;
            var hashCode = org_repo.split("").reduce(function(a, b) {
                a = ((a << 5) - a) + b.charCodeAt(0);
                return a & a;
              }, 0);

            var category_start = "";
            if (rank == 1) {
                html = "<tr><td>";
                // var category_link = "<a class='label-link' title='" + CATEGORY_DATA[row.category] + "' href='/?c=" + row.category + "'>" + row.category + "</a>";
                var category_link = "<h1><a class='label-link' title='" + CATEGORY_DATA[row.category] + "' href='/?c=" + row.category + "'>" + CATEGORY_DATA[row.category] + "</a></h1>";
                html += category_link + "<br />";
            }

            var asset_title = row._repopath + ", Score: " + row._pop_score + ", Stars per week: " + row._stars_per_week;
            var asset_url = "https://opengraph.githubassets.com/" + hashCode + "/" + org_repo; // + "?size=50";
            var asset_img = "<img src='" + asset_url + "' width='32%' alt='" + asset_title + "' />";
            var asset_link = "<a href='https://github.com/" + row._repopath + "' title='" + asset_title + "' alt='" + asset_title + "'>" + asset_img + "</a>";
            html += asset_link + "&nbsp;";

            if (rank == MAX_ASSETS) {
                html + "</td></tr>";
                $("#top_repos_table").append(html);
            }
        }
    });
});