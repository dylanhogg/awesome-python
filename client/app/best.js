const version = "v0.0.12";
const CATEGORY_COL = 0;  // 0-based
const TAG_COL = 8;

// TODO: load CATEGORY_DATA from url and sinc with app.js...
var CATEGORY_DATA = {
    '': 'Select category...',
    'all': 'All categories',
    'awesome': 'Awesome Lists',
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
    'ml-rl': 'ML - Reinforcement',
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
        var tbl = $("<table />").attr("id", "best_repos_table");
        $("#best_repos").append(tbl);

        var MAX_ASSETS = 3;
        var repos = data["data"];
        var td_html = "";
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

            // Row header
            if (rank == 1) {
                var href_title = "View more repos in category " + CATEGORY_DATA[row.category];
                header = "<tr><td colspan='3'>";
                header += "<hr /><h4><a title='" + href_title + "' href='/?c=" + row.category + "'>" + CATEGORY_DATA[row.category] + "</a></h4>";
                header += "</tr>";
                td_html = "<tr>";
                $("#best_repos_table").append(header);
            }

            // Row cells
            var asset_title = row._repopath + ", Score: " + row._pop_score + ", Stars per week: " + row._stars_per_week;
            var asset_url = "https://opengraph.githubassets.com/" + hashCode + "/" + org_repo; // + "?size=50";
            var asset_img = "<img src='" + asset_url + "' width='100%' alt='" + asset_title + "' />";
            var asset_link = "<a href='https://github.com/" + row._repopath + "' title='" + asset_title + "' alt='" + asset_title + "'>" + asset_img + "</a>";
            td_html += "<td>" + asset_link + "</td>"

            // Row footer
            if (rank == MAX_ASSETS) {
                // if (i<7) { alert(rank); }
                td_html += "</tr>";
                $("#best_repos_table").append(td_html);
            }
        }
    });
});