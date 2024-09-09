--require("user")
print("Hello from NEOVIM .config/nvim")

-- vim options
vim.opt.guicursor = ""

-- show line number
vim.opt.nu = true

-- allows neovim to access the system clipbloard
vim.opt.clipboard="unnamedplus"

-- allow the mouse in neovim
vim.opt.mouse = "a"

-- to get smartindent
vim.opt.tabstop = 4
vim.opt.softtabstop = 4
vim.opt.shiftwidth = 4
vim.opt.expandtab = true
vim.opt.smartindent = true

-- search word in neovim
vim.opt.hlsearch = false
vim.opt.incsearch = true

vim.opt.termguicolors = true

vim.opt.scrolloff = 10

vim.opt.updatetime = 50

vim.opt.colorcolumn = "80"

-- show tab
vim.opt.showtabline = 2

-- leader key
vim.g.mapleader = " "
-- LATEX VIEWER
vim.g.livepreview_engine = 'xelatex'

-- keymap settings
vim.keymap.set("n", "<leader>pv", vim.cmd.Ex)

-- deplacer des blocs de code
vim.keymap.set("v", "<C-Down>", ":m '>+1<CR>gv=gv")
vim.keymap.set("v", "<C-Up>", ":m '<-2<CR>gv=gv")

-- deplacer la recherche au mileu
vim.keymap.set("n", "n", "nzzzv")
vim.keymap.set("n", "N", "Nzzzv")

-- remap delete without copying
vim.keymap.set("x", "<leader>p", "\"_dP")

-- copy to system clipboard or vim
vim.keymap.set("n", "<leader>y", "\"+y")
vim.keymap.set("v", "<leader>y", "\"+y")
vim.keymap.set("n", "<leader>Y", "\"+Y")

-- replace all similar occurrences
vim.keymap.set("n", "<leader>s", [[:%s/\<<C-r><C-w>\>/<C-r><C-w>/gI<Left><Left><Left>]])
vim.keymap.set("n", "<leader>x", "<cmd>!chmod +x %<CR>", { silent = true })

-- Visual --
-- Stay in indent mode
vim.keymap.set("v", "<", "<gv")
vim.keymap.set("v", ">", ">gv")

-- launch floating terminal
vim.keymap.set("n", "<leader>t", ":FloatermNew<CR>" )

-- switch between header and source
vim.keymap.set("n", "<leader>h", ":Ouroboros<CR>" )

-- use cmake easily
vim.keymap.set("n", "<leader>cg", ":CMakeGenerate<CR>")
vim.keymap.set("n", "<leader>cb", ":CMakeBuild<CR>")
vim.keymap.set("n", "<leader>cc", ":CMakeClose<CR>")



-- configure lazy package manager
local lazypath = vim.fn.stdpath 'data' .. '/lazy/lazy.nvim'
if not vim.loop.fs_stat(lazypath) then
	vim.fn.system {
		'git',
		'clone',
		'--filter=blob:none',
		'https://github.com/folke/lazy.nvim.git',
		'--branch=stable', -- latest stable release
		lazypath,
	}
end
vim.opt.rtp:prepend(lazypath)

-- configure lazy setup
require('lazy').setup({
    {
        -- Theme inspired by Atom
        --'navarasu/onedark.nvim',
        'folke/tokyonight.nvim',
        priority = 1000,
        config = function()
            --vim.cmd.colorscheme 'onedark'
            vim.cmd.colorscheme 'tokyonight'
        end,
    },


    {
      "windwp/nvim-autopairs",
    },

    {
        "nvim-tree/nvim-tree.lua",
        dependencies = { "nvim-tree/nvim-web-devicons" },
    },

    {
        "goolord/alpha-nvim",
        event = "VimEnter",
        dependencies = { "nvim-tree/nvim-web-devicons" },
    },

    {
        "nvim-telescope/telescope.nvim",
        branch = "0.1.x",
        dependencies = {
            "nvim-lua/plenary.nvim",
            { "nvim-telescope/telescope-fzf-native.nvim", build = "make" },
        },
    },

    -- NOTE: This is where your plugins related to LSP can be installed.
    --  The configuration is done below. Search for lspconfig to find it below.
    {
        -- LSP Configuration & Plugins
        'neovim/nvim-lspconfig',
        dependencies = {
            -- Automatically install LSPs to stdpath for neovim
            { 'williamboman/mason.nvim', config = true },
            'williamboman/mason-lspconfig.nvim',

            -- Useful status updates for LSP
            -- NOTE: `opts = {}` is the same as calling `require('fidget').setup({})`
            { 'j-hui/fidget.nvim', tag = 'legacy', opts = {} },

            -- Additional lua configuration, makes nvim stuff amazing!
            'folke/neodev.nvim',
        },
    },


    {
        -- Autocompletion
        'hrsh7th/nvim-cmp',
        dependencies = {
            -- Snippet Engine & its associated nvim-cmp source
            'L3MON4D3/LuaSnip',
            'saadparwaiz1/cmp_luasnip',

            -- Adds LSP completion capabilities
            'hrsh7th/cmp-nvim-lsp',

            -- Adds a number of user-friendly snippets
            'rafamadriz/friendly-snippets',
        },
    },

    -- "gc" to comment visual regions/lines
    { 'numToStr/Comment.nvim', opts = {} },


    {
        -- Highlight, edit, and navigate code
        'nvim-treesitter/nvim-treesitter',
        dependencies = {
            'nvim-treesitter/nvim-treesitter-textobjects',
        },
        build = ':TSUpdate',
    },


    -- latex utils
    {
        'lervag/vimtex'
    },

    -- latex preview
--    {
--        'xuhdev/vim-latex-live-preview'
--    },

    -- cmake for vim plugin
    {
        "cdelledonne/vim-cmake"
    },

    -- float terminal below to run command
    {
        'voldikss/vim-floaterm'
    },


    -- switch between sources and headers
    {
        'jakemason/ouroboros',
        dependencies = { {'nvim-lua/plenary.nvim'} }
    },

})




-- configuration telescope
local telescope = require("telescope")
telescope.setup({
})

telescope.load_extension("fzf")

-- set keymaps
local keymap = vim.keymap -- for conciseness

keymap.set("n", "<leader>pf", "<cmd>Telescope find_files<cr>", { desc = "Fuzzy find files in cwd" }) -- find files within current working directory, respects .gitignore
keymap.set("n", "<leader>pr", "<cmd>Telescope oldfiles<cr>", { desc = "Fuzzy find recent files" }) -- find previously opened files
keymap.set("n", "<leader>ps", "<cmd>Telescope live_grep<cr>", { desc = "Find string in cwd" }) -- find string in current working directory as you type
keymap.set("n", "<leader>pc", "<cmd>Telescope grep_string<cr>", { desc = "Find string under cursor in cwd" }) -- find string under cursor in current working directory
keymap.set("n", "<leader>pb", "<cmd>Telescope buffers<cr>", { desc = "Show open buffers" }) -- list open buffers in current neovim instance
keymap.set("n", "<leader>hf", "<cmd>Telescope harpoon marks<cr>", { desc = "Show harpoon marks" }) -- show harpoon marks
keymap.set("n", "<leader>gc", "<cmd>Telescope git_commits<cr>", { desc = "Show git commits" }) -- list all git commits (use <cr> to checkout) ["gc" for git commits]
keymap.set("n", "<leader>gfc", "<cmd>Telescope git_bcommits<cr>", { desc = "Show git commits for current buffer" }) -- list git commits for current file/buffer (use <cr> to checkout) ["gfc" for git file commits]
keymap.set("n", "<leader>gb", "<cmd>Telescope git_branches<cr>", { desc = "Show git branches" }) -- list git branches (use <cr> to checkout) ["gb" for git branch]
keymap.set("n", "<leader>gs", "<cmd>Telescope git_status<cr>", { desc = "Show current git changes per file" }) -- list current changes per file with diff preview ["gs" for git status]



-- [[ Configure LSP ]]
local on_attach = function(_, bufnr)
  local nmap = function(keys, func, desc)
    if desc then
      desc = 'LSP: ' .. desc
    end

    vim.keymap.set('n', keys, func, { buffer = bufnr, desc = desc })
  end

  nmap('<leader>rn', vim.lsp.buf.rename, '[R]e[n]ame')
  nmap('<leader>ca', vim.lsp.buf.code_action, '[C]ode [A]ction')

  nmap('gd', vim.lsp.buf.definition, '[G]oto [D]efinition')
  nmap('gr', require('telescope.builtin').lsp_references, '[G]oto [R]eferences')
  nmap('gI', vim.lsp.buf.implementation, '[G]oto [I]mplementation')
  nmap('<leader>D', vim.lsp.buf.type_definition, 'Type [D]efinition')
  nmap('<leader>ds', require('telescope.builtin').lsp_document_symbols, '[D]ocument [S]ymbols')
  nmap('<leader>ws', require('telescope.builtin').lsp_dynamic_workspace_symbols, '[W]orkspace [S]ymbols')

  -- See `:help K` for why this keymap
  nmap('K', vim.lsp.buf.hover, 'Hover Documentation')
  nmap('<C-k>', vim.lsp.buf.signature_help, 'Signature Documentation')

  -- Lesser used LSP functionality
  nmap('gD', vim.lsp.buf.declaration, '[G]oto [D]eclaration')
  nmap('<leader>wa', vim.lsp.buf.add_workspace_folder, '[W]orkspace [A]dd Folder')
  nmap('<leader>wr', vim.lsp.buf.remove_workspace_folder, '[W]orkspace [R]emove Folder')
  nmap('<leader>wl', function()
    print(vim.inspect(vim.lsp.buf.list_workspace_folders()))
  end, '[W]orkspace [L]ist Folders')

  -- Create a command `:Format` local to the LSP buffer
  vim.api.nvim_buf_create_user_command(bufnr, 'Format', function(_)
    vim.lsp.buf.format()
  end, { desc = 'Format current buffer with LSP' })
end

-- Enable the following language servers
local servers = {
    texlab = {},
    cmake = {},
    bashls = {},
    clangd = {},
    pyright = {},
    lua_ls = {
        Lua = {
            workspace = { checkThirdParty = false },
            telemetry = { enable = false },
        },
    },
}

-- Setup neovim lua configuration
require('neodev').setup()

-- nvim-cmp supports additional completion capabilities, so broadcast that to servers
local capabilities = vim.lsp.protocol.make_client_capabilities()
capabilities = require('cmp_nvim_lsp').default_capabilities(capabilities)

-- Ensure the servers above are installed
local mason_lspconfig = require 'mason-lspconfig'

mason_lspconfig.setup {
  ensure_installed = vim.tbl_keys(servers),
}

mason_lspconfig.setup_handlers {
  function(server_name)
    require('lspconfig')[server_name].setup {
      capabilities = capabilities,
      on_attach = on_attach,
      settings = servers[server_name],
      filetypes = (servers[server_name] or {}).filetypes,
    }
  end
}

-- [[ Configure nvim-cmp ]]
-- See `:help cmp`
local cmp = require 'cmp'
local luasnip = require 'luasnip'
require('luasnip.loaders.from_vscode').lazy_load()
require("luasnip.loaders.from_lua").load({paths = "~/.config/nvim/snippets/"})

luasnip.config.setup {}

cmp.setup {
    snippet = {
        expand = function(args)
            luasnip.lsp_expand(args.body)
        end,
    },
    mapping = cmp.mapping.preset.insert {
        ['<C-n>'] = cmp.mapping.select_next_item(),
        ['<C-p>'] = cmp.mapping.select_prev_item(),
        ['<C-d>'] = cmp.mapping.scroll_docs(-4),
        ['<C-f>'] = cmp.mapping.scroll_docs(4),
        ['<C-Space>'] = cmp.mapping.complete {},
        ['<CR>'] = cmp.mapping.confirm {
            behavior = cmp.ConfirmBehavior.Replace,
            select = true,
        },
        ['<Tab>'] = cmp.mapping(function(fallback)
            if cmp.visible() then
                cmp.select_next_item()
            elseif luasnip.expand_or_locally_jumpable() then
                luasnip.expand_or_jump()
            else
                fallback()
            end
        end, { 'i', 's' }),
        ['<S-Tab>'] = cmp.mapping(function(fallback)
            if cmp.visible() then
                cmp.select_prev_item()
            elseif luasnip.locally_jumpable(-1) then
                luasnip.jump(-1)
            else
                fallback()
            end
        end, { 'i', 's' }),
    },
    sources = {
        { name = 'nvim_lsp' },
        { name = 'luasnip' },
    },
}



local nvimtree = require("nvim-tree")

-- recommended settings from nvim-tree documentation ces commandes bugs
--vim.g.loaded_netrw = 1
--vim.g.loaded_netrwPlugin = 1

-- change color for arrows in tree to light blue
vim.cmd([[ highlight NvimTreeIndentMarker guifg=#3FC5FF ]])

-- configure nvim-tree
nvimtree.setup({
    view = {
        width = 60,
    },
    -- change folder arrow icons
    renderer = {
        icons = {
            glyphs = {
                folder = {
                    arrow_closed = "", -- arrow when folder is closed
                    arrow_open = "", -- arrow when folder is open
                },
            },
        },
    },
    -- disable window_picker for
    -- explorer to work well with
    -- window splits
    actions = {
        open_file = {
            window_picker = {
                enable = false,
            },
        },
    },
    filters = {
        custom = { ".DS_Store" },
    },
    git = {
        ignore = false,
    },
})

-- set keymaps
keymap.set("n", "<leader>ee", "<cmd>NvimTreeToggle<CR>", { desc = "Toggle file explorer" }) -- toggle file explorer
keymap.set("n", "<leader>ef", "<cmd>NvimTreeFindFileToggle<CR>", { desc = "Toggle file explorer on current file" }) -- toggle file explorer on current file
keymap.set("n", "<leader>ec", "<cmd>NvimTreeCollapse<CR>", { desc = "Collapse file explorer" }) -- collapse file explorer
keymap.set("n", "<leader>er", "<cmd>NvimTreeRefresh<CR>", { desc = "Refresh file explorer" }) -- refresh file explorer



local alpha = require("alpha")
local dashboard = require("alpha.themes.dashboard")

-- Set header
dashboard.section.header.val = {
    "",
    " ███▄ ▄███▓ ▄▄▄     ▄▄▄█████▓ ██▀███   ██▓▒██   ██▒",
    "▓██▒▀█▀ ██▒▒████▄   ▓  ██▒ ▓▒▓██ ▒ ██▒▓██▒▒▒ █ █ ▒░",
    "▓██    ▓██░▒██  ▀█▄ ▒ ▓██░ ▒░▓██ ░▄█ ▒▒██▒░░  █   ░",
    "▒██    ▒██ ░██▄▄▄▄██░ ▓██▓ ░ ▒██▀▀█▄  ░██░ ░ █ █ ▒ ",
    "▒██▒   ░██▒ ▓█   ▓██▒ ▒██▒ ░ ░██▓ ▒██▒░██░▒██▒ ▒██▒",
    "░ ▒░   ░  ░ ▒▒   ▓▒█░ ▒ ░░   ░ ▒▓ ░▒▓░░▓  ▒▒ ░ ░▓ ░",
    "░  ░      ░  ▒   ▒▒ ░   ░      ░▒ ░ ▒░ ▒ ░░░   ░▒ ░",
    "░      ░     ░   ▒    ░        ░░   ░  ▒ ░ ░    ░  ",
    "       ░         ░  ░           ░      ░   ░    ░  ",
    "                                                   ",
}

-- Set menu
dashboard.section.buttons.val = {
    dashboard.button("e", "  > New File", "<cmd>ene<CR>"),
    dashboard.button("SPC ee", "  > Toggle file explorer", "<cmd>NvimTreeToggle<CR>"),
    dashboard.button("SPC ff", "󰱼 > Find File", "<cmd>Telescope find_files<CR>"),
    dashboard.button("SPC fs", "  > Find Word", "<cmd>Telescope live_grep<CR>"),
    dashboard.button("SPC wr", "󰁯  > Restore Session For Current Directory", "<cmd>SessionRestore<CR>"),
    dashboard.button("q", " > Quit NVIM", "<cmd>qa<CR>"),
}

-- Send config to alpha
alpha.setup(dashboard.opts)

-- Disable folding on alpha buffer
vim.cmd([[autocmd FileType alpha setlocal nofoldenable]])



require'nvim-treesitter.configs'.setup {
    -- A list of parser names, or "all"
    ensure_installed = { 'c', 'cpp', 'go', 'lua', 'python', 'rust', 'tsx', 'typescript', 'vimdoc', 'vim', 'cmake', 'latex' },

    -- Install parsers synchronously (only applied to `ensure_installed`)
    sync_install = false,

    -- Automatically install missing parsers when entering buffer
    -- Recommendation: set to false if you don't have `tree-sitter` CLI installed locally
    auto_install = true,

    highlight = {
        -- `false` will disable the whole extension
        enable = true,

        -- Instead of true it can also be a list of languages
        additional_vim_regex_highlighting = false,
    },
}


require'nvim-autopairs'.setup{}



-- -- Définition de la fonction personnalisée
-- function ReplaceParentheses(new_text)
--   vim.fn.setreg('a', new_text)
--   vim.cmd('%s/\\(([^)]*)\\)/\\=getreg(\'a\')/g')
-- end
--
-- -- Création de la commande personnalisée
-- vim.cmd('command! -nargs=1 ReplaceParentheses lua ReplaceParentheses(<args>)')


-- Définition de la fonction personnalisée
function ReplaceParentheses(new_text)
  local command = ':%s/(\\zs[^)]*\\ze)/' .. vim.fn.escape(new_text, '/') .. '/g'
  vim.cmd(command)
end

-- Création de la commande personnalisée
vim.cmd('command! -nargs=1 ReplaceParentheses lua ReplaceParentheses(<args>)')



-- -- Définition de la fonction personnalisée pour le remplacement dans la sélection visuelle
-- function ReplaceParenthesesVisual(new_text)
--   local selected_text = vim.fn.getreg('v')
--   local replaced_text = string.gsub(selected_text, '%b()', new_text)
--   vim.fn.setreg('v', replaced_text)
-- end
--
-- -- Création de la commande personnalisée pour le remplacement dans la sélection visuelle
-- vim.cmd('command! -nargs=1 ReplaceParenthesesVisual lua ReplaceParenthesesVisual(<args>)')


-- Définition de la fonction personnalisée pour le remplacement dans la sélection visuelle
function ReplaceParenthesesVisual(new_text)
  -- Récupérez la sélection visuelle
  local selected_text = vim.fn.getreg('v')

  -- Appliquez la substitution avec une expression régulière pour le texte entre parenthèses
  local replaced_text = string.gsub(selected_text, '(%b())', '(' .. new_text .. ')')

  -- Mettez à jour la sélection visuelle avec le texte modifié
  vim.fn.setreg('v', replaced_text)

  -- Réinitialisez la sélection visuelle
  vim.fn.feedkeys("gv")
end

-- Création de la commande personnalisée pour le remplacement dans la sélection visuelle
vim.cmd('command! -nargs=1 ReplaceParenthesesVisual :<,>lua ReplaceParenthesesVisual(<args>)')


-- vim.cmd([[command! Testoperator :'<,'>:s/document/testop]])
vim.cmd([[command! Testoperator :'<,'>:s/document/testop]])


vim.cmd([[command! -range Testoperator2 :<line1>,<line2>s/test/bout/g]])


-- Créez une commande personnalisée en Lua
vim.cmd([[command! -range -nargs=0 ReplaceBracesLua :<line1>,<line2>lua ReplaceBraces()]])

-- Fonction pour effectuer les substitutions en chaîne
function ReplaceBraces()
  local range = vim.fn.line("'<") .. "," .. vim.fn.line("'>")
  local cmd = "execute '" .. range .. "s/{/{{/g' | execute '" .. range .. "s/}/}}/g'"
  vim.fn.execute(cmd)
end


-- Fonction pour commenter les lignes contenant "import pdb"
function CommentPdbLines()
  local start_line = 1
  local end_line = vim.fn.line('$')  -- Obtient le nombre total de lignes dans le fichier
  for line_num = start_line, end_line do
    local line_content = vim.fn.getline(line_num)
    if string.match(line_content, "import%s+pdb") then
      -- Commente la ligne si elle contient "import pdb"
      vim.fn.setline(line_num, "# " .. line_content)
    end
  end
end

-- Créer une commande personnalisée : :CommentPdb
vim.api.nvim_create_user_command('CommentPdb', CommentPdbLines, {})
