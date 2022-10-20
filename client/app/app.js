var version = "v0.0.8";
var CATEGORY_COL = 9;

function getUrlParams() {
    // Ref: https://stackoverflow.com/questions/4656843/get-querystring-from-url-using-jquery/4656873#4656873
    var vars = [], hash;
    var hashes = window.location.href.slice(window.location.href.indexOf('?') + 1).split('&');
    for(var i = 0; i < hashes.length; i++) {
        hash = hashes[i].split('=');
        vars.push(hash[0]);
        vars[hash[0]] = hash[1];
    }
    return vars;
}

function getUrlQuery() {
    try {
        var params = getUrlParams();
        if ("q" in params) { return decodeURI(params["q"]); } else { return ""; }
    } catch(err) {
        return "";
    }
}

function getUrlCategoryFilter() {
    try {
        var params = getUrlParams();
        if ("c" in params) { return decodeURI(params["c"]); } else { return ""; }
    } catch(err) {
        return "";
    }
}

$(document).keydown(function(e) {
    if (e.keyCode == 39) {  // Right arrow
        $("#table").DataTable().page("next").draw("page");
    } else if (e.keyCode == 37) {  // Left arrow
        $("#table").DataTable().page("previous").draw("page");
    } else if (e.keyCode == 27) {  // Escape
        $('#sidenav').width("0px");
    }
});

$(document).ready( function () {
    var ajax_url = './github_data.min.json';
    // var ajax_url = 'https://crazy-awesome-python-api.infocruncher.com/github_data.min.json';
    if (location.hostname === "localhost" || location.hostname === "127.0.0.1") {
        // Use local testing json data
        ajax_url = '/github_data.json';
    }

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

    var initialSearchTerm = getUrlQuery();
    var table = $("#table").DataTable( {
        ajax: {
            url: ajax_url,
            dataSrc: 'data'
        },
        responsive: true,
        order: [[ 1, "desc" ]],
        paging: true,
        lengthChange: true,
        lengthMenu: [[10, 50, 100, -1], [10, 50, 100, "All"]],
        pageLength: 10,
        search: {
           search: initialSearchTerm,
        },
        language: {
            searchPlaceholder: "Search",
            search: "",
          },
        // dom: 'lfrtip',  // Default. https://datatables.net/reference/option/dom
        dom: 'frtilp',
        columns: [
          { data: null,
            title: "Name",
            render: function(data, type, row, meta) {
                var repoUrl = "<a href='" + row.githuburl + "'>" + row._reponame.toLowerCase() + "</a>";
                return repoUrl;
             }
           },
           { data: "_stars", title: "Stars&nbsp;<img src='img/star.png' class='github-img' />", className: "text-nowrap", render: $.fn.dataTable.render.number(',', '.', 0) },
           { data: "_stars_per_week", title: "Stars<br />per&nbsp;week",
            render: function(data, type, row, meta) { return data > 10 ? data.toFixed(0) : data.toFixed(1); }
           },
           { data: "_description", title: "Description",
             render: function(data, type, row, meta) { return "<div class='text-wrap description-column'>" + data + "</div>"; }
           },
           { data: null,
            title: "Links",
            render: function(data, type, row, meta) {
                var repoUrl = "<a href='" + row.githuburl + "' target='_blank'>" + "<img src='img/repo.png' class='github-img'></img></a>&nbsp;<a href='" + row.githuburl + "'>" + row._reponame.toLowerCase() + "</a>";
                var orgUrl = "<br /><a href='https://github.com/" + row._organization + "' target='_blank'>" + "<img src='img/org.png' class='github-img'></img></a>&nbsp;<a href='https://github.com/" + row._organization + "'>" + row._organization.toLowerCase() + "</a>";
                var homepageUrl = "";
                try { homepageUrl = "<br /><a href='" + row._homepage + "' target='_blank'><img src='img/web.png' class='web-img'></img></a>&nbsp;<a href='" + row._homepage + "'>" + new URL(row._homepage).hostname + "</a>"; } catch { }
                return repoUrl + orgUrl + homepageUrl;
             }
           },
           { data: "_forks", title: "Forks&nbsp;<img src='img/fork.png' class='github-img' />", className: "text-nowrap", render: $.fn.dataTable.render.number(',', '.', 0) },
           { data: "_age_weeks", title: "Age in&nbsp;weeks",
            render: function(data, type, row, meta) { return data.toFixed(0); }
           },
           { data: "_created_at", title: "Created&nbsp;<img src='img/clock.png' class='github-img' />",
            className: "text-nowrap",
            render: function(data, type, row, meta) { return new Date(data).toISOString().split('T')[0]; }
           },
           { data: "_updated_at", title: "Updated&nbsp;<img src='img/clock.png' class='github-img' />",
            className: "text-nowrap",
            render: function(data, type, row, meta) { return new Date(data).toISOString().split('T')[0]; }
           },
           { data: "category", title: "Category" },
           { data: "_topics", title: "Tags",
            render: function(data, type, row, meta) { return data.slice(0, 3).join(", "); }
           },
           { data: "_readme_localurl", title: "Docs",
            orderable: false,
            render: function(data, type, row, meta) {
                if (data.length > 0) {
                    var url = "/data/" + data + "";
                    return "<img src='img/github.png' alt='info' title='View install and GitHub info' class='modal-ajax info-img' href='#' data-localurl='"+url+"' data-ext='.html' data-title='' data-replace-lf='false'></img>";
                } else {
                    return "";
                }
            }
           },
          //          { data: "_requirements_localurls", title: "Requirements",
//            render: function(data, type, row, meta) {
//                if (data.length > 0) {
//                    var links = "";
//                    for (var i=0; i<data.length; i++) {
//                        var filename = data[i];
//                        var url = "/data/" + filename + "";
//                        var title = "unknown";
//                        if (filename.indexOf("requirements.txt") > 0) {
//                            title = "requirements.txt";
//                        } else if (filename.indexOf("setup.py") > 0) {
//                            title = "setup.py";
//                        } else if (filename.indexOf("pyproject.toml") > 0) {
//                            title = "pyproject.toml";
//                        }
//                        links = links + "<a class='modal-ajax' href='#' data-localurl='"+url+"' data-ext='' data-title='"+title+"' data-replace-lf='true'>"+title+"</a><br />";
//                    }
//                    return links;
//                } else {
//                    return "";
//                }
//
//            }
//          },
        ],
    });

//    table.on("search.dt", function () {
//        table_search = table.search();
//        console.log("table_search:" + table_search);
////        if (table_search.startsWith("c:")) {
////            category_filter = table_search.split(":")[1]
////            console.log("category_filter:" + category_filter);
////            table.search("")  // Clear full-table search
////                 .columns(CATEGORY_COL)
////                 .search(category_filter);
////        }
//
////        else {
////            table.columns(CATEGORY_COL).search("");  // Clear col search
////        }
//    });

    $("#categories").change(function(){
        category_filter = $("#categories").val()
        console.log("category_filter:" + category_filter);
        console.log(table);
        table
            // .search("")  // Clear full-table search
            .columns(CATEGORY_COL)
            .search("^"+category_filter+"$", true, false)  // regex search
            //.search(category_filter)
            .draw();
    });

    var initialCategoryFilter = getUrlCategoryFilter();
    if (initialCategoryFilter.length > 0) {
        $("#categories").val(initialCategoryFilter).change();
    }

    $('#table').on('click', '.modal-ajax', function(e) {
        var localurl = $(this).data('localurl') + $(this).data('ext');
        e.preventDefault();

        $.ajax({
           type: "GET",
           url: localurl,
           title: $(this).data('title'),
           replace_lf: $(this).data('replace-lf'),
           success: function(content)
           {
                if (this.replace_lf) {
                    content = content.replace(/\n/g, '<br />');
                }
                var html = "<div class='modal'>";
                if (this.title.length > 0) {
                    html = html + "<b>" + this.title + "</b><br /><br />";
                }
                html = html + content + "</div>";
                $(html).appendTo("#container").modal();
           },
           error: function(html)
           {
                console.log("ERROR getting localurl: " + localurl);
           },
        });

        return false;
    });
});



