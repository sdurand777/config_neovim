local ls = require("luasnip") --{{{
local s = ls.s --> snippet
local i = ls.i --> insert node
local t = ls.t --> text node

local d = ls.dynamic_node
local c = ls.choice_node
local f = ls.function_node
local sn = ls.snippet_node

local fmt = require("luasnip.extras.fmt").fmt
local rep = require("luasnip.extras").rep

local snippets, autosnippets = {}, {} --}}}

local group = vim.api.nvim_create_augroup("all files", { clear = true })
local file_pattern = "*.*"

local function cs(trigger, nodes, opts) --{{{
	local snippet = s(trigger, nodes)
	local target_table = snippets

	local pattern = file_pattern
	local keymaps = {}

	if opts ~= nil then
		-- check for custom pattern
		if opts.pattern then
			pattern = opts.pattern
		end

		-- if opts is a string
		if type(opts) == "string" then
			if opts == "auto" then
				target_table = autosnippets
			else
				table.insert(keymaps, { "i", opts })
			end
		end

		-- if opts is a table
		if opts ~= nil and type(opts) == "table" then
			for _, keymap in ipairs(opts) do
				if type(keymap) == "string" then
					table.insert(keymaps, { "i", keymap })
				else
					table.insert(keymaps, keymap)
				end
			end
		end

		-- set autocmd for each keymap
		if opts ~= "auto" then
			for _, keymap in ipairs(keymaps) do
				vim.api.nvim_create_autocmd("BufEnter", {
					pattern = pattern,
					group = group,
					callback = function()
						vim.keymap.set(keymap[1], keymap[2], function()
							ls.snip_expand(snippet)
						end, { noremap = true, silent = true, buffer = true })
					end,
				})
			end
		end
	end

	table.insert(target_table, snippet) -- insert snippet into appropriate table
end --}}}

-- Start Refactoring --


cs("snipscript", fmt( -- snippet type
[[
local ls = require("luasnip") --{{{{{{
local s = ls.s --> snippet
local i = ls.i --> insert node
local t = ls.t --> text node

local d = ls.dynamic_node
local c = ls.choice_node
local f = ls.function_node
local sn = ls.snippet_node

local fmt = require("luasnip.extras.fmt").fmt
local rep = require("luasnip.extras").rep

local snippets, autosnippets = {{}}, {{}} --}}}}}}

local group = vim.api.nvim_create_augroup("{}", {{ clear = true }})
local file_pattern = "*.{}"

local function cs(trigger, nodes, opts) --{{{{{{
	local snippet = s(trigger, nodes)
	local target_table = snippets

	local pattern = file_pattern
	local keymaps = {{}}

	if opts ~= nil then
		-- check for custom pattern
		if opts.pattern then
			pattern = opts.pattern
		end

		-- if opts is a string
		if type(opts) == "string" then
			if opts == "auto" then
				target_table = autosnippets
			else
				table.insert(keymaps, {{ "i", opts }})
			end
		end

		-- if opts is a table
		if opts ~= nil and type(opts) == "table" then
			for _, keymap in ipairs(opts) do
				if type(keymap) == "string" then
					table.insert(keymaps, {{ "i", keymap }})
				else
					table.insert(keymaps, keymap)
				end
			end
		end

		-- set autocmd for each keymap
		if opts ~= "auto" then
			for _, keymap in ipairs(keymaps) do
				vim.api.nvim_create_autocmd("BufEnter", {{
					pattern = pattern,
					group = group,
					callback = function()
						vim.keymap.set(keymap[1], keymap[2], function()
							ls.snip_expand(snippet)
						end, {{ noremap = true, silent = true, buffer = true }})
					end,
				}})
			end
		end
	end

	table.insert(target_table, snippet) -- insert snippet into appropriate table
end --}}}}}}


-- Ecrire ses snippets lua on peut utiliser le snipet luasnippet 






-- Tutorial Snippets go here --

-- End Refactoring --

return snippets, autosnippets

]], {
    i(1, "decrire le type de fichier"),
    i(2, "taper extension du fichier"),
}))


cs( -- [luaSnippet] LuaSnippet{{{
	"luaSnippetsimple",
	fmt(
		[=[
cs("{}", fmt( -- {}
[[{}]], {{{}}}))]=],
		{
			i(1, ""),
			i(2, "Description"),
			i(3, ""),
			i(4, ""),
		}
	)
) --}}}



cs("ivmtemp", fmt( -- ivm pdf template
[=[
\documentclass[]{{paper}}
\usepackage[a4paper, total={{7in, 9in}}]{{geometry}}
\usepackage{{longfbox}}
\renewcommand\familydefault{{\sfdefault}}
% Useful packages
\usepackage{{amsmath,amsfonts}}
\usepackage[colorlinks=true, allcolors=blue]{{hyperref}}
% header
\usepackage{{fancyhdr}}
\pagestyle{{fancy}}
\fancyhead{{}}
\fancyhead[R]{{IVM Technologies - Projet NARVAL~- Page~\thepage }} 
\renewcommand{{\headrulewidth}}{{0.2pt}} % no line in header area
% footer
\fancyfoot{{}} % clear all footer fields
% page number in "outer" position of footer line
\fancyfoot[C]{{\scriptsize \indent All our business relations are regulated by our general conditions of sale, communicable on request. This document is the property of IVM TECHNOLO  GIES.
Anyone holding a copy acknowledges being bound by an obligation of secrecy and undertakes not to reproduce it, to communicate it to third parties in whole or in part,
or to use it for personal purposes, without written authorization from IVM TECHNOLOGIES.}}
\fboxset{{padding-left=6pt,padding-right=6pt,padding-top=12pt,padding-bottom=12pt}}%

\usepackage{{tikz}}

\usepackage[french]{{babel}}
\usepackage{{datetime}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}

\title{{\huge Compte Rendu \today : {} }}
\author{{Auteur : {} }}
\subtitle{{\huge {} }}

\usepackage{{fontsize}}
\changefontsize[17]{{17}}
  
\begin{{document}}
  
\maketitle
  
\begin{{tikzpicture}}[remember picture,overlay]
     \node[anchor=north east,inner sep=0pt] at (current page.north east)
              {{\includegraphics[scale=0.5]{{IVM}}}};
\end{{tikzpicture}}
  
\section{{ }}
\end{{document}}

]=], {
  i(1, "title"),
  i(2, "author"),
  i(3, "subtitle"),
  }))


cs("cmakesimp", fmt( -- un cmake simple pour tester
[[
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(hello)
add_executable(hello main.cpp)
{}
]],
{
    i(1,""),
}))


cs("cmakeaddlocalelib", fmt( -- simple commande pour ajouter une librairie locale
[[
add_subdirectory(src/{})
include_directories(${{PROJECT_SOURCE_DIR}}/include/{})
{}
]], {
    i(1,"chemin vers les fichiers cpp de la lib"),
    i(2,"chemin vers les headers de la lib"),
    i(3,""),
}))


cs("texsimple", fmt( -- simple document latex
[[
\documentclass[12pt]{{article}}
\begin{{document}}
Test
\end{{document}}
{}
]], {
    i(1,""),
  }))


cs("bashrc_gcc_command", fmt( -- bashrc gcc command 
[[
# set alias gcc include paths
alias gccinclude='gcc -xc++ -E -v -'

PATH1="/usr/local/include/eigen3"
PATH2="/usr/local/include/igl"
PATH3="/usr/local/include/GLFW"
PATH4="/usr/local/include/vtk-9.2"
PATH5="/usr/include/opencv4"

export CPLUS_INCLUDE_PATH=$PATH1:$PATH2:$PATH3:$PATH4:$PATH5
]], {
  }))


cs("bashrc_cuda_include", fmt( -- bashrc cuda include command 
[[
export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
]], {
  }))


cs("bashrc_comp_command", fmt( -- bashrc comp command 
[[
alias comp='mkdir build && cd build && cmake .. && make'  
]], {
  }))



 cs("OpenCV_cmake", fmt( -- opencv cmake 
 [[
 find_package(OpenCV REQUIRED)
 include_directories(${{OpenCV_INCLUDE_DIRS}})
 add_executable(main main.cpp)
 target_link_libraries(main ${{OpenCV_LIBS}})
 ]], {
   }))


 cs("Pangolin_cmake", fmt( -- Pangolin cmake 
 [[
 find_package(Pangolin REQUIRED)
 include_directories(${{Pangolin_INCLUDE_DIRS}})
 add_executable(main main.cpp)
 target_link_libraries(main ${{Pangolin_LIBRARIES}})
 ]], {
   }))
-- Tutorial Snippets go here --

-- End Refactoring --

return snippets, autosnippets
