var version = "v0.0.2";

$(document).ready( function () {
    $("#table").DataTable( {
        ajax: {
            url: '/github_data.json',  // Local testing
            // url: 'https://crazy-awesome-python-api.infocruncher.com/github_data.json',
            dataSrc: 'data'
        },
        order: [[ 5, "desc" ]],
        columns: [
          { data: "_readme_localurl", title: "Info",
            render: function(data, type, row, meta) {
                var url = "/data/" +  data + "";
                return "<a class='modal-ajax' href='#' data-localurl='"+url+"'>info</a>";
            }
          },
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

    var showdown_converter = new showdown.Converter();
    $('#table').on('click', '.modal-ajax', function(e) {
        var localurl = $(this).data('localurl');
        console.log(localurl);
        e.preventDefault();

        $.get(localurl, function(content) {
            var html = "";
            if (localurl.toLowerCase().endsWith(".md")) {
                html = showdown_converter.makeHtml(content);
                html = "<div class='modal'>"
                    + "TEMP: I've been processed by showdown from "+localurl+"<br />"
                    + html
                    + "</div>";
                $(html).appendTo("#container").modal();
            } else {
                var htmlurl = localurl+ ".html";
                $.get(htmlurl, function(html) {
                    html = "<div class='modal'>"
                        + "TEMP: I'm direct from " + htmlurl + "<br />"
                        + html
                        + "</div>";
                    $(html).appendTo("#container").modal();
                });
            }
        });

        return false;
    });
});



