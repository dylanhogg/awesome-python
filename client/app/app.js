const version = "v0.0.11";
const CATEGORY_COL = 7;  // 0-based
const TAG_COL = 8;

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

function deleteUrlQueryKey(paramKey) {
    try {
        var href = new URL(location.href);
        href.searchParams.delete(paramKey);
        window.history.pushState("", "unused", href);
    } catch(err) { }
}

function setUrlQueryValue(paramKey, paramVal) {
    try {
        var href = new URL(location.href);
        href.searchParams.set(paramKey, paramVal);
        window.history.pushState("", "unused", href);
    } catch(err) { }
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
    if (e.keyCode == 39 || e.keyCode == 34) {  // Right arrow or page down
        $("#table").DataTable().page("next").draw("page");
    } else if (e.keyCode == 37 || e.keyCode == 33) {  // Left arrow or page up
        $("#table").DataTable().page("previous").draw("page");
    } else if (e.keyCode == 27) {  // Escape
        $('#sidenav').width("0px");
    }
});

// Category filter dropdown
$(document).on("preInit.dt", function (e, settings) {
    var data = {
        '': 'Select category...',
        'all': 'All categories',
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
    var select = $('<select name="category_filter" id="category_filter" class="form-select-sm form-select-sm category_filter" />');
    for(var val in data) {
        $('<option />', {value: val, text: data[val]}).appendTo(select);
    }
    select.appendTo('div.dataTables_filter');

    select.change(function(){
        category_filter = $("#category_filter").val();
        var table = $("#table").DataTable();
        if (category_filter == "" || category_filter == "all") {
            deleteUrlQueryKey("c");
            table
                //.search("")  // Clear full-table search
                .columns(CATEGORY_COL)
                .search("")
                .draw();
        } else {
            setUrlQueryValue("c", category_filter);
            table
                // .search("")  // Clear full-table search
                .columns(CATEGORY_COL)
                //.search(category_filter, true, false)  // too broad, e.g. ml gets ml-dl also
                .search("^"+category_filter+"$", true, false)  // regex search over plain text (ensure filter: true for <a>...</a> columns)
                .draw();
        }
    });

    var initialCategoryFilter = getUrlCategoryFilter();
    if (initialCategoryFilter.length > 0) {
        $("#category_filter").val(initialCategoryFilter).change();
    }
});

// Tag filter dropdown (WIP)
//$(document).on("preInit.dt", function (e, settings) {
//    var select = $('<select name="tag_filter" id="tag_filter" class="form-select-sm form-select-sm tag_filter">' +
//                   '<option value="">All Tags</option></select>');
//    select.appendTo('div.dataTables_filter');
//
//    $.getJSON("github_tags_data.json", function( data ) {
//        $.each( data, function( key, val ) {
//            $('<option />', {value: key, text: val}).appendTo(select);
//        });
//
//        select.change(function(){
//            tag_filter = $("#tag_filter").val()
//            var table = $("#table").DataTable();
//            if (tag_filter == "") {
//                table
//                    //.search("")  // Clear full-table search
//                    .columns(TAG_COL)
//                    .search("")
//                    .draw();
//            } else {
//                table
//                    // .search("")  // Clear full-table search
//                    .columns(TAG_COL)
//                    .search(tag_filter)  // TODO: review match single tag
//                    .draw();
//            }
//        });
//    });
//});

$(document).ready( function () {
    var ajax_url = './github_data.ui.min.json?v1.0';

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
    var description_max_strlen = 99;
    var topic_max_strlen = 30;
    var topic_max_count = 3;

    var table = $("#table").DataTable( {
        ajax: {
            url: ajax_url,
            dataSrc: 'data'
        },
        // searchDelay: 100,
        deferRender: true,
        responsive: true,
        order: [[ 1, "desc" ]],
        paging: true,
        pagingType: "full",  // https://datatables.net/reference/option/pagingType
        lengthChange: true,
        lengthMenu: [[5, 10, 50, -1], [5, 10, 50, "All"]],
        pageLength: 5,
        search: {
           search: initialSearchTerm,
        },
        filter: true,  // Enables anchor wrapped category filtering
        language: {
            searchPlaceholder: "Search...",
            search: "",
          },
        // dom: 'lfrtip',  // Default. https://datatables.net/reference/option/dom
        // dom: 'frtilp',  // Awesome Python - Classic
        dom: 'frtpil',  // Awesome Python - Page above info, needs some custom css for alignment
        // dom: '<"top"fp>rt<"bottom"il><"clear">',  // Awesome Python - Top Buttons
        columns: [
          { data: null,
            title: "Name",
            render: function(data, type, row, meta) {
                var repoUrl = "<a href='" + row.githuburl + "'>" + row._reponame.toLowerCase() + "</a>";
                return repoUrl;
             }
           },
           { data: "_pop_score", title: "Score&nbsp;🔥", className: "text-nowrap", render: $.fn.dataTable.render.number(',', '.', 0) },
           { data: "_stars", title: "Stars&nbsp;⭐", className: "text-nowrap", render: $.fn.dataTable.render.number(',', '.', 0) },
           { data: "_description", title: "Description",
             render: function(data, type, row, meta) {
                if(data.length > description_max_strlen) {
                    return "<div class='text-wrap description-column'>" + data.substr(0, description_max_strlen) + "…</div>";
                } else {
                    return "<div class='text-wrap description-column'>" + data + "</div>";
                }
             }
           },
           { data: "_stars_per_week", title: "Stars<br />per&nbsp;week",
            render: function(data, type, row, meta) { return data > 10 ? data.toFixed(0) : data.toFixed(1); }
           },
           { data: "_age_weeks", title: "Age in&nbsp;weeks",
            render: function(data, type, row, meta) { return data.toFixed(0); }
           },

//           { data: "_pop_contributor_count", title: "contributors", className: "text-nowrap", render: $.fn.dataTable.render.number(',', '.', 0) },
//           { data: "_pop_commit_frequency", title: "commits<br />frequency", className: "text-nowrap", render: $.fn.dataTable.render.number(',', '.', 0) },
//           { data: "_pop_issue_count", title: "issues", className: "text-nowrap", render: $.fn.dataTable.render.number(',', '.', 0) },
//           { data: "_pop_comment_count", title: "comments", className: "text-nowrap", render: $.fn.dataTable.render.number(',', '.', 0) },
//           { data: "_pop_contributor_orgs_len", title: "contributor_orgs", className: "text-nowrap", render: $.fn.dataTable.render.number(',', '.', 0) },
//           { data: "_pop_updated_issues_count", title: "updated_issues_count", className: "text-nowrap", render: $.fn.dataTable.render.number(',', '.', 0) },
//           { data: "_pop_closed_issues_count", title: "closed_issues_count", className: "text-nowrap", render: $.fn.dataTable.render.number(',', '.', 0) },
//           { data: "_pop_created_since_days", title: "created_since_days", className: "text-nowrap", render: $.fn.dataTable.render.number(',', '.', 0) },
//           { data: "_pop_updated_since_days", title: "updated_since_days", className: "text-nowrap", render: $.fn.dataTable.render.number(',', '.', 0) },
//           { data: "_pop_recent_releases_count", title: "recent_releases_count", className: "text-nowrap", render: $.fn.dataTable.render.number(',', '.', 0) },
//           { data: "_pop_recent_releases_estimated_tags", title: "recent_releases_estimated_tags", className: "text-nowrap", render: $.fn.dataTable.render.number(',', '.', 0) },
//           { data: "_pop_recent_releases_adjusted_count", title: "recent_releases_adjusted_count", className: "text-nowrap", render: $.fn.dataTable.render.number(',', '.', 0) },
//           { data: "_pop_comment_count_lookback_days", title: "comment_count_lookback_days", className: "text-nowrap", render: $.fn.dataTable.render.number(',', '.', 0) },
//           { data: "_pop_comment_frequency", title: "comment_frequency", className: "text-nowrap", render: $.fn.dataTable.render.number(',', '.', 0) },

           { data: null,
            title: "Links",
            render: function(data, type, row, meta) {
                var repoUrl = "<a href='" + row.githuburl + "' target='_blank'>" + "<img src='img/repo.png' width='16' height='16' alt='repo' title='View GitHub repo' class='github-img'></img></a>&nbsp;<a href='" + row.githuburl + "'>" + row._reponame.toLowerCase() + "</a>";
                var orgUrl = "<br /><a href='https://github.com/" + row._organization + "' target='_blank'>" + "<img src='img/org.png' width='16' height='16' alt='organisation' title='View GitHub organisation' class='github-img'></img></a>&nbsp;<a href='https://github.com/" + row._organization + "'>" + row._organization.toLowerCase() + "</a>";
                var homepageUrl = "";
                try { homepageUrl = "<br /><a href='" + row._homepage + "' target='_blank'><img src='img/web16.png' width='16' height='16' alt='homepage' title='View homepage' class='web-img'></img></a>&nbsp;<a href='" + row._homepage + "'>" + new URL(row._homepage).hostname + "</a>"; } catch { }
                return "<div class='text-wrap links-column'>" + repoUrl + orgUrl + homepageUrl + "</div>";
             }
           },

//           { data: "_forks", title: "Forks&nbsp;<img src='img/fork.png' class='github-img' />", className: "text-nowrap", render: $.fn.dataTable.render.number(',', '.', 0) },
//           { data: "_created_at", title: "Created&nbsp;<img src='img/clock.png' class='github-img' />",
//            className: "text-nowrap",
//            render: function(data, type, row, meta) { return new Date(data).toISOString().split('T')[0]; }
//           },
//           { data: "_updated_at", title: "Updated&nbsp;<img src='img/clock.png' class='github-img' />",
//            className: "text-nowrap",
//            render: function(data, type, row, meta) { return new Date(data).toISOString().split('T')[0]; }
//           },

           { data: "category", title: "Category"
             ,render: function(data, type, row, meta) {
                // return data;
                return "<a class='label-link' href='/?c=" + data + "'>" + data + "</a>";
             }
           },
           { data: "_topics", title: "Tags",
            // render: function(data, type, row, meta) { return data.slice(0, 3).join(", "); }
            render: function(data, type, row, meta) {
                if (data.length == 0) { return ""; }
                var tags = data.slice(0, topic_max_count);
                return tags.map(item => {
                    var short_item = item;
                    if (short_item.length > topic_max_strlen) {
                        short_item = short_item.substr(0, topic_max_strlen);
                    }
                    // return "<a class='label-link' href='https://github.com/topics/" + item + "?l=python'>" + item + "</a> ";;
                    return "<a class='label-link' href='/?q=" + short_item + "'>" + short_item + "</a> ";
                }).join(" ");
            }
           },

//           { data: "_readme_localurl", title: "Docs",
//            orderable: false,
//            render: function(data, type, row, meta) {
//                if (data.length > 0) {
//                    var readme_localurl = "/data/" + data + "";
//                    return "<img src='img/github16.png' width='16' height='16' alt='docs' title='View repo readme documentation' class='modal-ajax info-img' href='#' data-localurl='"+readme_localurl+"' data-ext='.html' data-title='' data-replace-lf='false'></img>";
//                } else {
//                    return "";
//                }
//            }
//           },

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
//            }
//          },
        ],  // end of columns
    });

    table.on('search.dt', function () {
        if (table.search() == "") {
            // Reset "q" url param (if exists) on cleared search box
            deleteUrlQueryKey("q");
        }
    });

    // Model for displaying _readme_localurl pop-up docs (deprecated)
    // js dependency:  <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/jquery-modal/0.9.2/jquery.modal.min.js"></script>
    // css dependency: <link rel="stylesheet" type="text/css" href="https://cdnjs.cloudflare.com/ajax/libs/jquery-modal/0.9.2/jquery.modal.min.css"/>
//    $('#table').on('click', '.modal-ajax', function(e) {
//        var localurl = $(this).data('localurl') + $(this).data('ext');
//        e.preventDefault();
//
//        $.ajax({
//           type: "GET",
//           url: localurl,
//           title: $(this).data('title'),
//           replace_lf: $(this).data('replace-lf'),
//           success: function(content)
//           {
//                if (this.replace_lf) {
//                    content = content.replace(/\n/g, '<br />');
//                }
//                var html = "<div class='modal'>";
//                if (this.title.length > 0) {
//                    html = html + "<b>" + this.title + "</b><br /><br />";
//                }
//                html = html + content + "</div>";
//                $(html).appendTo("#container").modal();
//           },
//           error: function(html)
//           {
//                console.log("ERROR getting localurl: " + localurl);
//           },
//        });
//        return false;
//    });
});



