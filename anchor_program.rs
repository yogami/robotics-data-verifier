use anchor_lang::prelude::*;
use anchor_spl::token::{self, Mint, Token, TokenAccount, Transfer};

declare_id!("DataAttest1111111111111111111111111111111111");

#[program]
pub mod robotics_data_attestation {
    use super::*;

    pub fn initialize_vault(ctx: Context<InitializeVault>) -> Result<()> {
        let vault_state = &mut ctx.accounts.vault_state;
        vault_state.authority = ctx.accounts.authority.key();
        vault_state.total_staked = 0;
        Ok(())
    }

    pub fn stake(ctx: Context<Stake>, amount: u64) -> Result<()> {
        let cpi_accounts = Transfer {
            from: ctx.accounts.provider_token_account.to_account_info(),
            to: ctx.accounts.vault_token_account.to_account_info(),
            authority: ctx.accounts.provider.to_account_info(),
        };
        let cpi_program = ctx.accounts.token_program.to_account_info();
        let cpi_ctx = CpiContext::new(cpi_program, cpi_accounts);
        token::transfer(cpi_ctx, amount)?;

        let provider_state = &mut ctx.accounts.provider_state;
        provider_state.staked_amount += amount;
        provider_state.provider = ctx.accounts.provider.key();
        
        let vault_state = &mut ctx.accounts.vault_state;
        vault_state.total_staked += amount;

        msg!("Staked {} tokens.", amount);
        Ok(())
    }

    pub fn submit_report_and_slash(
        ctx: Context<SubmitReport>, 
        dataset_id: String, 
        report_hash: String, 
        entropy_score: f64,
        anomaly_rate: f64
    ) -> Result<()> {
        // Only the authorized ML engine Oracle can submit reports
        require!(ctx.accounts.oracle_authority.key() == ctx.accounts.vault_state.authority, ErrorCode::UnauthorizedOracle);

        let attestation = &mut ctx.accounts.attestation;
        attestation.provider = ctx.accounts.provider.key();
        attestation.dataset_id = dataset_id;
        attestation.report_hash = report_hash;
        attestation.timestamp = Clock::get()?.unix_timestamp;

        // Slashing Logic based on ML metrics
        let mut slash_percentage = 0.0;
        
        // If kinematic entropy is extremely low (padding data)
        if entropy_score < 0.5 {
            slash_percentage += 0.5; // Slash 50%
        }
        
        // If isolation forest detects massive anomalies (erratic garbage data)
        if anomaly_rate > 0.10 {
            slash_percentage += 0.5; // Slash 50%
        }

        if slash_percentage > 0.0 {
            let provider_state = &mut ctx.accounts.provider_state;
            let slash_amount = (provider_state.staked_amount as f64 * slash_percentage) as u64;
            
            provider_state.staked_amount -= slash_amount;
            
            msg!("SLASHED: Provider submitted low-quality data. Slashed {} tokens.", slash_amount);
            // In a real implementation, tokens would be transferred to a burn address or treasury.
        } else {
            msg!("Data Quality Report Anchored. Hash: {} (Quality Verified)", attestation.report_hash);
        }

        Ok(())
    }
}

#[derive(Accounts)]
pub struct InitializeVault<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + 32 + 8,
        seeds = [b"vault_state"],
        bump
    )]
    pub vault_state: Account<'info, VaultState>,
    #[account(mut)]
    pub authority: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct Stake<'info> {
    #[account(mut)]
    pub provider: Signer<'info>,
    #[account(mut)]
    pub provider_token_account: Account<'info, TokenAccount>,
    #[account(mut)]
    pub vault_token_account: Account<'info, TokenAccount>,
    #[account(mut, seeds = [b"vault_state"], bump)]
    pub vault_state: Account<'info, VaultState>,
    #[account(
        init_if_needed,
        payer = provider,
        space = 8 + 32 + 8,
        seeds = [b"provider_state", provider.key().as_ref()],
        bump
    )]
    pub provider_state: Account<'info, ProviderState>,
    pub token_program: Program<'info, Token>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
#[instruction(dataset_id: String)]
pub struct SubmitReport<'info> {
    #[account(mut)]
    pub oracle_authority: Signer<'info>,
    /// CHECK: provider account is safe
    pub provider: AccountInfo<'info>,
    #[account(mut, seeds = [b"vault_state"], bump)]
    pub vault_state: Account<'info, VaultState>,
    #[account(mut, seeds = [b"provider_state", provider.key().as_ref()], bump)]
    pub provider_state: Account<'info, ProviderState>,
    #[account(
        init,
        payer = oracle_authority,
        space = 8 + 32 + 4 + dataset_id.len() + 4 + 64 + 8,
        seeds = [b"attestation", dataset_id.as_bytes()],
        bump
    )]
    pub attestation: Account<'info, DataAttestation>,
    pub system_program: Program<'info, System>,
}

#[account]
pub struct VaultState {
    pub authority: Pubkey,
    pub total_staked: u64,
}

#[account]
pub struct ProviderState {
    pub provider: Pubkey,
    pub staked_amount: u64,
}

#[account]
pub struct DataAttestation {
    pub provider: Pubkey,
    pub dataset_id: String,
    pub report_hash: String,
    pub timestamp: i64,
}

#[error_code]
pub enum ErrorCode {
    #[msg("Unauthorized Oracle. Only the verifiable data quality engine can submit reports.")]
    UnauthorizedOracle,
}
