var version = "v0.0.3";

$(document).ready( function () {
    $("#table").DataTable( {
        ajax: {
            // url: '/github_data.json',  // Local testing
            url: 'https://crazy-awesome-python-api.infocruncher.com/github_data.json',
            dataSrc: 'data'
        },
        order: [[ 5, "desc" ]],
        columns: [
          { data: "_readme_localurl", title: "",
            orderable: false,
            render: function(data, type, row, meta) {
                if (data.length > 0) {
                    var url = "/data/" + data + "";
                    return "<img src='img/info.png' alt='info' class='modal-ajax' href='#' data-localurl='"+url+"' data-ext='.html' data-title='' data-replace-lf='false'></img>";
                } else {
                    return "";
                }
            }
          },
//          { data: "_requirements_localurls", title: "Requrements",
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
          { data: "category", title: "Category" },
          { data: "_description", title: "Description" },
          { data: "_repopath", title: "Github",
            render: function(data, type, row, meta) { return "<a href='https://github.com/" + data + "'>" + data + "</a>"; }
          },
          { data: "_homepage", title: "Homepage",
              render: function(data, type, row, meta)
              {
                try { return "<a href='" + data + "'>" + new URL(data).hostname + "</a>"; }
                catch { return ""; }
              }
          },
//          { data: "_topics", title: "Tags",
//            render: function(data, type, row, meta) { return data.join(", "); }
//          },
          { data: "_stars", title: "Stars", className: "text-nowrap", render: $.fn.dataTable.render.number(',', '.', 0) },
          { data: "_stars_per_week", title: "Stars\nper&nbsp;week",
            render: function(data, type, row, meta) { return data > 10 ? data.toFixed(0) : data.toFixed(1); }
          },
          { data: "_forks", title: "Forks", className: "text-nowrap", render: $.fn.dataTable.render.number(',', '.', 0) },
          { data: "_created_at", title: "Created",
            className: "text-nowrap",
            render: function(data, type, row, meta) { return new Date(data).toISOString().split('T')[0]; }
          },
          { data: "_age_weeks", title: "Age in&nbsp;weeks",
            render: function(data, type, row, meta) { return data.toFixed(0); }
          },
        ],
        paging: false,
    });

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



